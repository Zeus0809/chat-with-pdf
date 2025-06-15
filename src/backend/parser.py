import pymupdf as pd
import pymupdf4llm as pd4llm
from src.backend import structure as st
import pprint
from typing import List

# build and retrieve full text from a pymupdf text block
def get_full_text(text_block: dict) -> str:
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

# retrieve font sizes from pymupdf text block, save if all text is the same size
def get_block_font_size(text_block: dict) -> int:
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

def build_page_content(page: pd.Page) -> List[st.ContentBlock]:
    """
    Builds a list of content blocks (different classes based on content) from a pymupdf page object.
    Data is extracted as a dict.
    """
    blocks = page.get_text("dict").get("blocks")
    content = []
    for block in blocks:
        block_keys = block.keys()
        if "lines" in block_keys:
            # build full text from block + check font size
            full_text = get_full_text(block)
            font_size = get_block_font_size(block)
            # create text object
            text_block = st.TextBlock(pos=block.get("bbox"),
                               block_index=block.get("number"),
                               blocks_on_page=len(blocks),
                               page=page.number + 1,
                               text=full_text,
                               font_size=font_size)
            content.append(text_block)
        elif "image" in block_keys:
            # create image object
            image_block = st.ImageBlock(pos=block.get("bbox"),
                                        block_index=block.get("number"),
                                        blocks_on_page=len(blocks),
                                        page=page.number + 1,
                                        blob=block.get("image"))
            content.append(image_block)
        else:
            content.append("404: unknown type")
    return content

def get_doc_markdown(doc: pd.Document) -> str:
    """
    Returns a markdown representation of the document content.
    """
    content = pd4llm.to_markdown(doc=doc, show_progress=True)
    print(f"\n###-Markdown-Start-###\n{content}\n###-Markdown-End-###\n")
    return content



