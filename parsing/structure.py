

class ContentBlock():
    """
    Base class for storing parsed PDF content in a structured format. Individual classes should inherit from this depending on content type.
    """

    def __init__(self, pos:tuple, block_index:int, blocks_per_page:int, page:int):

        # Type asserts
        try:
            assert isinstance(pos, tuple)
        except Exception as e:
            raise TypeError("Position of content block must be a tuple")
        try:
            assert isinstance(block_index, int)
        except Exception as e:
            raise TypeError("Index of content block must be an integer")
        try:
            assert isinstance(blocks_per_page, int)
        except Exception as e:
            raise TypeError("Total block count per page must be an integer")
        try:
            assert isinstance(page, int)
        except Exception as e:
            raise TypeError("Block page number must be an integer")

        # Block position (float)
        self.x0 = pos[0]
        self.y0 = pos[1]
        self.x1 = pos[2]
        self.y1 = pos[3]
        # Block size (float)
        self.width = pos[2] - pos[0]
        self.height = pos[3] - pos[1]
        # Block metadata
        self.page_position = (block_index+1, blocks_per_page)
        self.page = page

    # to string
    def __str__(self):
        return f"Content block {self.page_position[0]} out of {self.page_position[1]} on page {self.page}"
    
