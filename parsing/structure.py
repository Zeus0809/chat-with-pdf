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
        self.x0 = pos[0]
        self.y0 = pos[1]
        self.x1 = pos[2]
        self.y1 = pos[3]
        # Block size (float)
        self.width = pos[2] - pos[0]
        self.height = pos[3] - pos[1]
        # Block metadata
        self.page_position = (block_index+1, blocks_on_page)
        self.page = page

    # to string
    def __str__(self):
        return f"Content block {self.page_position[0]} out of {self.page_position[1]} on page {self.page}"

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

        assert isinstance(text, str), "Text must be a string."
        assert (font_size is None or isinstance(font_size, int)), "Font size must be an integer, or None if not applicable (block text has multiple font sizes)."

        # set text attributes
        self.text = text
        self.font_size = font_size

    def __str__(self):
        return f"Text block starting with '{self.text[:7]}' and ending with '{self.text[len(self.text)-7:]}'."

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
        return f"Image block of size {self.size} pt"



