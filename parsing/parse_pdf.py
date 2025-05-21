import pymupdf
import pprint
import structure as st

doc = pymupdf.open("./data/SOR-2001-269.pdf")
# get all pages
pages = [page for page in doc]
# get blocks from page
first_page = pages[0].get_text("dict")
#pprint.pprint(first_page)
first_page_blocks = first_page.get("blocks")

# Build content blocks for page 1
page_content = []
for block in first_page_blocks:
    
    # Create appropriate objects based on block type
    if "lines" in block.keys():
        # build full text from block + check font size
        full_text = ""
        font_sizes = []
        for line in block.get("lines"):
            for span in line.get("spans"):
                full_text += (span.get("text") + " ")
                font_sizes.append(span.get("size"))
        # see what's up with font size
        font_size = int(font_sizes[0]) if all(s == font_sizes[0] for s in font_sizes) else None
        # create text object
        text_block = st.TextBlock(pos=block.get("bbox"),
                                     block_index=block.get("number"),
                                     blocks_on_page=len(first_page_blocks),
                                     page=pages[0].number+1,
                                     text=full_text,
                                     font_size=font_size)
        page_content.append(text_block)
    elif "image" in block.keys():
        # create image object
        image_block = st.ImageBlock(pos=block.get("bbox"),
                                     block_index=block.get("number"),
                                     blocks_on_page=len(first_page_blocks),
                                     page=pages[0].number+1,
                                     blob=block.get("image"))
        page_content.append(image_block)
    else:
        page_content.append("Block of unknown type")

for i, c in enumerate(page_content):
    print(f"Block #{i+1}:", c)
    try:
        print("\t", c.text)
    except:
        print("\t", "This is not a text block.")


