class ContentBlock():
    """
    Base class for storing parsed PDF content in a structured format. Individual classes should inherit from this depending on content type.
    """

    def __init__(self, pos:tuple, block_index:int, blocks_on_page:int, page:int):

        # Type asserts
        assert isinstance(pos, tuple), "Position of content block must be a tuple."
        assert isinstance(block_index, int), "Index of content block must be an integer."
        assert isinstance(blocks_on_page, int), "Total block count per page must be an integer."
        assert isinstance(page, int), "Block page number must be an integer."

        # Block position (float)
        self.x0 = round(pos[0], 2)
        self.y0 = round(pos[1], 2)
        self.x1 = round(pos[2], 2)
        self.y1 = round(pos[3], 2)
        # Block size (float)
        self.width = round(self.x1 - self.x0, 2)
        self.height = round(self.y1 - self.y0, 2)
        # Block metadata
        self.page_position = (block_index+1, blocks_on_page)
        self.page = page

class TextBlock(ContentBlock):
    """
    A content block that contains text. Meant for paragraphs, headings, sub-headings, etc.
    """
    def __init__(self, pos:tuple,
                 block_index:int,
                 blocks_on_page:int,
                 page:int,
                 text:str, font_size:int):
        # initialize parent
        super().__init__(pos, block_index, blocks_on_page, page)

        assert isinstance(text, str), f"Text must be a string, instead got {type(text)}."
        assert isinstance(font_size, int), f"Font size must be an integer, instead got {type(font_size)}. If block has multiple font sizes, expect -1."

        # set text attributes
        self.text = text
        self.font_size = font_size

    def __str__(self):
        result = f"Text block at: ({self.x0}, {self.y0}) - ({self.x1}, {self.y1})\n"
        result += f"Of size: {self.width} x {self.height} pt\n"
        result += f"Order: {self.page_position[0]} of {self.page_position[1]} on page {self.page}\n"
        result += f"Font size: {self.font_size} pt\n"
        result += f"-------------------Text:-------------------\n{self.text}\n-------------------------------------------\n"
        return result

class ImageBlock(ContentBlock):
    """
    A content block that contains an image and its data. Meant to pass image data to a VLM for description and later vectorization.
    """
    def __init__(self, pos:tuple,
                 block_index:int,
                 blocks_on_page:int,
                 page:int,
                 blob:bytes):
        
        # initialize parent
        super().__init__(pos, block_index, blocks_on_page, page)

        assert isinstance(blob, bytes), "Image blob must be of type 'bytes'."

        # set image attributes
        self.blob = blob
        self.size = self.width * self.height

    def __str__(self):
        result = f"Image block at: ({self.x0}, {self.y0}) - ({self.x1}, {self.y1})\n"
        result += f"Of size: {self.width} x {self.height} pt\n"
        result += f"Order: {self.page_position[0]} of {self.page_position[1]} on page {self.page}\n"
        return result



