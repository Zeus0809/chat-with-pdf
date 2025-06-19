import pymupdf as pd
import pymupdf4llm as pd4llm
from src.backend import structure as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import List, Optional
import pprint, time, io
from PIL import Image

BLIP_MODEL_PATH = "./local_models/vision/blip_large"

class PDFParser:
    """
    Parser class for all parsing operations on the loaded PDF. Provides methods to parse the document into various formats and extract metadata.
    """
    def __init__(self):
        self.pdf_markdown: Optional[str] = None  # markdown representation of the document
        self.pdf_blocks: Optional[List[List[st.ContentBlock]]] = None  # structured document handle 
        self._blip_processor: BlipProcessor = None
        self._blip_model: BlipForConditionalGeneration = None

    def _load_blip_model(self) -> None:
        """
        Loads the BLIP model for image captioning. Loading is done on-demand.
        """
        if self._blip_processor is None:
            self._blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_PATH)
            self._blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_PATH)
        else:
            print("--BLIP model already loaded--")

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
        return full_text.strip()

    def _get_block_font_size(self, text_block: dict) -> int:
        """
        Retrieves font size from a pymupdf text block. If block has multiple font sizes, returns 0, otherwise returns font size.
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

    def _build_page_content(self, page: pd.Page) -> List[st.ContentBlock]:
        """
        Builds a list of content blocks (different classes based on content) from a pymupdf page object.
        Data is extracted as a dict.
        """
        assert isinstance(page, pd.Page), f"'page' must be a pymupdf Page object. Instead got {type(page)}."
        blocks = page.get_text("dict").get("blocks")
        content = []

        for block in blocks:
            block_keys = block.keys()
            if "lines" in block_keys:
                # build full text from block + check font size
                full_text = self._get_full_text(block)
                font_size = self._get_block_font_size(block)
                # create text object
                text_block = st.TextBlock(pos=block.get("bbox"),
                                block_index=block.get("number"),
                                blocks_on_page=len(blocks),
                                page=page.number + 1,
                                text=full_text,
                                font_size=font_size)
                content.append(text_block)

            elif "image" in block_keys:
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
                content.append("404: unknown type")

        return content

    def parse_to_markdown(self, doc: pd.Document) -> str:
        """
        Returns a markdown representation of the document content.
        """
        assert isinstance(doc, pd.Document), f"'doc' must be a pymupdf Document object. Instead got {type(doc)}."
        start = time.time()
        self.pdf_markdown = pd4llm.to_markdown(doc=doc, show_progress=True)
        print(f"\n###-Markdown-Start-###\n{self.pdf_markdown}\n###-Markdown-End-(parsed in {round(time.time()-start, 2)}s)###\n")

    def parse_to_blocks(self, doc: pd.Document) -> None:
        """
        Parses the document into structured content blocks from structure.py.
        """
        assert isinstance(doc, pd.Document), f"'doc' must be a pymupdf Document object. Instead got {type(doc)}."
        start = time.time()
        self.pdf_blocks = []
        for page in doc:
            page_content = self._build_page_content(page)
            self.pdf_blocks.append(page_content)
        time_taken = round(time.time() - start, 2)
        # temporary
        # self.debug_parsed_blocks(time_taken)

    def debug_parsed_blocks(self, time_taken: float) -> None:
        """
        Prints the parsed blocks to the console for debugging purposes.
        """
        assert self.pdf_blocks is not None, "No blocks parsed yet. Please call PDFParser.parse_to_blocks() first."
        block_count = 0
        print("\n###-Parsed-Blocks-Start-###")
        for page in self.pdf_blocks:
            for block in page:
                print(block)
                block_count += 1
        print(f"###-Parsed-Blocks-End-(parsed {block_count} blocks in {time_taken}s)###\n")

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
        assert isinstance(self._blip_processor, BlipProcessor), "BLIP model not loaded. Please call _load_blip_model() first."
        assert isinstance(blob, bytes), "Image blob must be of type 'bytes'."
        start = time.time()
        image = Image.open(io.BytesIO(blob)).convert("RGB")
        inputs = self._blip_processor(image, return_tensors="pt")
        outputs = self._blip_model.generate(**inputs)
        caption = self._blip_processor.decode(outputs[0], skip_special_tokens=True)
        print(f"--BLIP caption generated in {round(time.time() - start, 2)}s:--")
        # print(f"{caption}\n")
        return caption
