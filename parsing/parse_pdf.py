import pymupdf
import pprint
from structure import ContentBlock

doc = pymupdf.open("./data/SOR-2001-269.pdf")
# get all pages
pages = [page for page in doc]
# get blocks from page
first_page = pages[0].get_text("dict")
first_page_blocks = first_page.get("blocks")

# Build content blocks for page 1
page_content = []
for block in first_page_blocks:
    content_block = ContentBlock(pos=block.get("bbox"),
                                 block_index=block.get("number"),
                                 blocks_per_page=len(first_page_blocks),
                                 page=pages[0].number+1)
    page_content.append(content_block)

for b in page_content:
    print(b)



