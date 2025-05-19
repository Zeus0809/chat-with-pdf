import pymupdf
import pprint

doc = pymupdf.open("./data/SOR-2001-269.pdf")

page_0 = doc[0].get_text("dict")
pprint.pprint(page_0)

# get blocks from page
blocks = page_0.get("blocks")
print("Blocks on page: ", len(blocks))
print()

for n, block in enumerate(blocks):
    print(f"--- block # {n} ---")

    lines = block.get("lines")
    if lines == None:
        image = block.get("image")
        block_type = "image"
    else:
        block_type = "text"

    print("Block type: ", block_type)
    if block_type == "text":
        print("Number of lines: ", len(lines))
        print("Text: ", ' '.join([line["spans"][0]["text"] for line in lines]))
    else:
        print("Image length: ", len(image))
        print("File format: ", block["ext"])


# 
