import pymupdf as pd
import pymupdf4llm as pd4llm
from src.backend import structure as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import List, Optional
import pprint, time, io, string, re
from PIL import Image

BLIP_MODEL_PATH = "./local_models/vision/blip_large"

# PyMuPDF font style flags
PLAIN_TEXT = 4
ITALIC_TEXT = 6
BOLD_TEXT = 20

# Bullet symbols
BULLET_SYMBOLS = ('•', '-', '*', '◦')

# Numbered list patterns (1-99)
NUMBERED_PATTERNS = (
    tuple(str(i) for i in range(1, 100)) +           # 1, 2, 3...
    tuple(f"{i}." for i in range(1, 100)) +          # 1., 2., 3...
    tuple(f"{i})" for i in range(1, 100)) +          # 1), 2), 3)...
    tuple(f"({i})" for i in range(1, 100))           # (1), (2), (3)...
)

# Lettered list patterns (a-z)
LETTERED_PATTERNS = (
    tuple(string.ascii_lowercase) +                   # a, b, c...
    tuple(f"{letter}." for letter in string.ascii_lowercase) +    # a., b., c...
    tuple(f"{letter})" for letter in string.ascii_lowercase) +    # a), b), c)...
    tuple(f"({letter})" for letter in string.ascii_lowercase)     # (a), (b), (c)...
)

# Combined list item characters
LIST_ITEM_CHARS = BULLET_SYMBOLS + NUMBERED_PATTERNS + LETTERED_PATTERNS

MAX_BULLET_NUM = 50 # max number for numbered lists to be considered as such
SUB_HEADING_MAX_WORDS = 5  # max words in a sub-heading to be considered as such

class PDFParser:
    """
    Parser class for all parsing operations on the loaded PDF. Provides methods to parse the document into various formats and extract metadata.
    Meant to construct a rich representation of the document content to be passed to LlamaIndex.
    """
    def __init__(self):
        self.pdf_chunks: Optional[List[dict]] = None  # chunked representation of the document for LlamaIndex embedding, built from pdf_blocks
        self.pdf_blocks: Optional[List[List[st.ContentBlock]]] = None  # structured document handle
        self._content_type_analysis: Optional[dict] = None # a map that relates font sizes, their frequency, and corresponding content type
        self._pdf_char_count: int = 0
        self._blip_processor: BlipProcessor = None
        self._blip_model: BlipForConditionalGeneration = None

    @staticmethod
    def _count_words(text: str) -> int:
            """
            Count words in text, handling PDF extraction artifacts.
            """
            if not text or not text.strip():
                return 0
            # Remove extra whitespace and normalize
            cleaned_text = re.sub(r'\s+', ' ', text.strip())
            # Count word-like sequences
            words = re.findall(r'\b\w+\b', cleaned_text)
            return len(words)

    def _load_blip_model(self) -> None:
        """
        Loads the BLIP model for image captioning. Loading is done on-demand.
        """
        if self._blip_processor is None:
            self._blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_PATH)
            self._blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_PATH)
        else:
            print("--BLIP model already loaded--")
        assert isinstance(self._blip_processor, BlipProcessor), f"BLIP processor failed to load. The attribute is of type {type(self._blip_processor)} instead of BlipProcessor."
        assert isinstance(self._blip_model, BlipForConditionalGeneration), f"BLIP model failed to load. The attribute is of type {type(self._blip_model)} instead of BlipForConditionalGeneration."

    def _get_full_text(self, text_block: dict) -> str:
        """
        Builds a full text string from a pymupdf text block.
        """
        full_text = ""
        for line in text_block.get("lines"):
            assert 'spans' in line, f"Line missing 'spans' key: {pprint.pprint(line)}"
            for span in line.get("spans"):
                assert 'text' in span, f"Span missing 'text' key: {pprint.pprint(span)}"
                full_text += (span.get("text") + " ")
                # print(f"--***--Span text: {span.get('text')}; Font size: {span.get('size', 0)}--***--")     
        return full_text.strip()

    def _analyze_content_types(self, doc: pd.Document) -> None:
        """
        Analyzes the document to determine the frequency of font sizes and use it to determine what content types they likely represent.
        """
        start = time.time()
        font_size_counts = {}

        for page in doc:
            blocks = page.get_text("dict", sort=True).get("blocks")
            for block in blocks:
                if "lines" in block:
                    for line in block.get("lines"):
                        assert 'spans' in line, f"Line missing 'spans' key: {pprint.pprint(line)}"
                        for span in line.get("spans"):
                            assert 'size' in span, f"Span missing 'size' key: {pprint.pprint(span)}"
                            size = int(span.get("size"))
                            text_length = len(span.get("text", ""))
                            # Count the frequency of each font size
                            font_size_counts[size] = font_size_counts.get(size, 0) + text_length
                            self._pdf_char_count += text_length

        # Build the font size analysis map using the frequency of each size
        self._content_type_analysis = {}
        for size, count in font_size_counts.items():
            frequency = round((count / self._pdf_char_count) * 100 , 2)
            self._content_type_analysis[size] = {"frequency": frequency, "content_type": "TBD"}

        # Classify content types based on frequency and update the map
        self._assign_content_types()

        print(f"--Content type analysis completed in {round(time.time() - start, 2)}s: \n\t{self._content_type_analysis}--")

    def _assign_content_types(self) -> None:
        """
        Assigns content types to font sizes based on custom rules:
            - Body text: the most frequent font size
            - Footnote: the smallest font size smaller than body text
            - Heading: the largest font size larger than body text
            - Sub-heading: all other sizes larger than body text
            - Other: all other sizes smaller than body text
        """
        assert len(self._content_type_analysis.keys()) > 0, f"Content type analysis map failed to build, contains 0 key-value pairs."
        
        # Find body text (the one with highest frequency)
        body_text_size = max(self._content_type_analysis, key=lambda k: self._content_type_analysis[k]["frequency"])
        self._content_type_analysis[body_text_size]["content_type"] = "body_text"

        # Classify other sizes
        for size in self._content_type_analysis.keys():
            if size == body_text_size:
                continue
            elif size < body_text_size:
                # Smallest size = footnote, others are 'other'
                smallest_size = min([s for s in self._content_type_analysis.keys() if s < body_text_size])
                if size == smallest_size:
                    self._content_type_analysis[size]["content_type"] = "footnote"
                else:
                    self._content_type_analysis[size]["content_type"] = "other"
            elif size > body_text_size:
                # Largest size = heading, others = sub-heading'
                largest_size = max([s for s in self._content_type_analysis.keys() if s > body_text_size])
                if size == largest_size:
                    self._content_type_analysis[size]["content_type"] = "heading"
                else:
                    self._content_type_analysis[size]["content_type"] = "sub-heading"
            
    def _get_block_font_size(self, text_block: dict) -> int:
        """
        Retrieves font size from a pymupdf text block. If block has multiple font sizes, returns -1, otherwise returns font size.
        """
        font_sizes = [] 
        for line in text_block.get("lines"):
            assert 'spans' in line, f"Line missing 'spans' key: {pprint.pprint(line)}"
            for span in line.get("spans"):
                assert 'size' in span, f"Span missing 'size' key: {pprint.pprint(span)}"
                # as soon as we hit a size we haven't seen before, break and return -1 because this block has multiple font sizes
                curr_size = int(span.get("size"))
                if curr_size not in font_sizes and len(font_sizes) > 0:
                    return -1
                font_sizes.append(curr_size)
        return font_sizes[0]

    def _get_block_font_styles(self, text_block: dict) -> set:
        """
        Retrieves font styles from a pymupdf text block. Returns a set of styles.
        """
        styles = set()
        for line in text_block.get("lines"):
            assert 'spans' in line, f"Line missing 'spans' key: {pprint.pprint(line)}"
            for span in line.get("spans"):
                flag = span.get("flags", PLAIN_TEXT) # default to plain text if not present
                
                if flag == PLAIN_TEXT:
                    styles.add("plain")
                elif flag == ITALIC_TEXT:
                    styles.add("italic")
                elif flag == BOLD_TEXT:
                    styles.add("bold")
                else:
                    styles.add("unknown")
        
        return styles          

    def _get_content_type(self, block_text: str, block_font_styles: set, block_font_size: int) -> str:
        """
        Determines the content type of a pymupdf text block based on custom rules.
        """
        assert isinstance(block_text, str), f"'block_text' must be a string. Instead got {type(block_text)}."
        assert isinstance(block_font_styles, set), f"'font_styles' must be a set. Instead got {type(block_font_styles)}."
        assert isinstance(block_font_size, int), f"'font_size' must be an integer. Instead got {type(block_font_size)}."

        # List items: Check for bullet points or numbered lists
        if (block_text.startswith(LIST_ITEM_CHARS) or 
            any(block_text.startswith(f"{i}.") or block_text.startswith(f"{i})") for i in range(1, MAX_BULLET_NUM))):
            return "list_item"
        
        if block_font_size == -1 or len(block_font_styles) > 1:
            return "mixed_content"
        
        # Classify blocks into headings, sub-headings, body text, footnotes, and other based on content type analysis
        for font_size in self._content_type_analysis.keys():
            if block_font_size == font_size:

                # handle case when sub-headings are as big as body_text but short and bold (e.g. **Required Information**, **Sub-category 'Corrosive'**)
                if (self._content_type_analysis[font_size]["content_type"] == "body_text"
                    and self._count_words(block_text) <= SUB_HEADING_MAX_WORDS
                    and "bold" in block_font_styles):
                        return "sub-heading"
                
                # handle image captions: italicized text appearing immediately before or after image

                else:
                    return self._content_type_analysis[font_size]["content_type"] 

    def _build_page_blocks(self, page: pd.Page) -> List[st.ContentBlock]:
        """
        Builds a list of content blocks (different classes based on content) from a pymupdf page object.
        Data is extracted as a dict.
        """
        assert isinstance(page, pd.Page), f"'page' must be a pymupdf Page object. Instead got {type(page)}."
        blocks = page.get_text("dict", sort=True).get("blocks")
        content = []

        for block in blocks:
            block_keys = block.keys()
            if "lines" in block_keys:
                # build full text from block + check font size
                full_text = self._get_full_text(block)
                font_styles = self._get_block_font_styles(block)
                font_size = self._get_block_font_size(block)
                content_type = self._get_content_type(full_text, font_styles, font_size)
                # create text object
                text_block = st.TextBlock(pos=block.get("bbox"),
                                block_index=block.get("number"),
                                blocks_on_page=len(blocks),
                                page=page.number + 1,
                                text=full_text,
                                font_size=font_size,
                                font_styles=font_styles,
                                content_type=content_type)
                content.append(text_block)

            elif "image" in block_keys:
                # lazy-load BLIP model
                self._load_blip_model()
                # run image captioning
                caption = self.image_to_text(block.get("image"))
                # create image object
                image_block = st.ImageBlock(pos=block.get("bbox"),
                                            block_index=block.get("number"),
                                            blocks_on_page=len(blocks),
                                            page=page.number + 1,
                                            caption=caption)
                content.append(image_block)
            else:
                content.append("404: unknown block type")

        return content

    def build_rich_chunked_content(self) -> None:
        """
        Produces a list of content chunks with separate metadata for LlamaIndex Document creation, stores it in self.pdf_chunks.
        This approach keeps embeddings clean while preserving rich metadata.
        """
        assert self.pdf_blocks is not None, "No blocks parsed yet. Please call PDFParser.parse_to_blocks() first."
        
        self.pdf_chunks = []
        for page in self.pdf_blocks:
            for block in page:
                assert isinstance(block, st.ContentBlock), f"Block must be a ContentBlock instance. Instead got {type(block)}."
                self.pdf_chunks.append(block.generate_chunk())

    def parse_to_blocks(self, doc: pd.Document) -> None:
        """
        Parses the document into structured content blocks from structure.py, stored in self.pdf_blocks.
        """
        assert isinstance(doc, pd.Document), f"'doc' must be a pymupdf Document object. Instead got {type(doc)}."

        # run font size analysis first to create content types -> they will be used in _build_page_blocks()
        self._analyze_content_types(doc)

        start = time.time()
        self.pdf_blocks = []
        for page in doc:
            page_content = self._build_page_blocks(page)
            self.pdf_blocks.append(page_content)
        time_taken = round(time.time() - start, 2)
        print(f"--Parsed document into blocks in {time_taken}s--")

    def debug_parsed_blocks(self) -> None:
        """
        Returns string representation of parsed blocks for debugging purposes.
        """
        assert self.pdf_blocks is not None, "No blocks parsed yet. Please call PDFParser.parse_to_blocks() first."
        
        block_count = 0
        header = "###-Parsed-Blocks-Start-###\n"
        body = ""
        for page in self.pdf_blocks:
            for block in page:
                body += str(block)
                block_count += 1
        footer = f"\n###-Parsed-Blocks-End-({block_count})###"

        return header + body + footer

    def debug_chunks(self) -> str:
        """
        Return pretty-printed representation of the chunked content for debugging purposes.
        """
        assert self.pdf_chunks is not None, "No chunks created yet. Please call PDFParser.build_chunked_content() first."
        
        header = "###-Chunked-Content-Start-###\n"
        body = ""
        for chunk in self.pdf_chunks:
            assert isinstance(chunk, dict), f"Chunk must be a dictionary. Instead got {type(chunk)}."
            body += pprint.pformat(chunk) + "\n"
        footer = "\n###-Chunked-Content-End###"

        return header + body + footer
        
    def clear_parsed_content(self) -> None:
        """
        Clears the parsed content to prepare for a new document.
        """
        self.pdf_markdown = None
        self.pdf_blocks = None
        print("--Parsed content cleared!--")

    def image_to_text(self, blob: bytes) -> str:
        """
        Uses the BLIP model to generate a caption for an image blob.
        """
        assert isinstance(blob, bytes), "Image blob must be of type 'bytes'."
        start = time.time()
        image = Image.open(io.BytesIO(blob)).convert("RGB")
        inputs = self._blip_processor(image, return_tensors="pt")
        outputs = self._blip_model.generate(**inputs)
        caption = self._blip_processor.decode(outputs[0], skip_special_tokens=True)
        print(f"--BLIP caption generated in {round(time.time() - start, 2)}s:--")
        assert isinstance(caption, str), f"BLIP caption must be a string, instead model produced {type(caption)}."
        return caption
