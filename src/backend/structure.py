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

    def _get_metadata(self) -> dict:
        """
        Generates metadata for content block. Children should customize this method.
        """
        return {
            'page_number': self.page,
            'coordinates_on_page': [self.x0, self.y0, self.x1, self.y1]
            # 'reading_order': self.page_position[0],
        }

    def generate_chunk(self) -> dict:
        """
        Generates a chunk of content with metadata for LlamaIndex embedding.
        This method should be overridden by child classes.
        """
        raise NotImplementedError("This method should be implemented in child classes.")

class TextBlock(ContentBlock):
    """
    A content block that contains text. Meant for paragraphs, headings, sub-headings, etc.
    """
    def __init__(self, pos:tuple,
                 block_index:int,
                 blocks_on_page:int,
                 page:int,
                 text:str,
                 font_size:int,
                 font_styles:set,
                 content_type:str):
        # initialize parent
        super().__init__(pos, block_index, blocks_on_page, page)

        assert isinstance(text, str), f"Text must be a string, instead got {type(text)}."
        assert isinstance(font_styles, set), f"Font styles must be a set, instead got {type(font_styles)}."
        assert isinstance(font_size, int), f"Font size must be an integer, instead got {type(font_size)}. If block has multiple font sizes, expect -1."
        assert isinstance(content_type, str), f"Content type must be a string, instead got {type(content_type)}."

        # set text attributes
        self.text = text.strip()
        self.font_size = font_size
        self.font_styles = font_styles
        self.content_type = content_type

    def _get_clean_content(self) -> str:
        """
        Returns clean semantic text content optimized for embeddings.
        """
        if self.content_type == "heading":
            return f"Section: {self.text}"
        elif self.content_type == "sub-heading":
            return f"Subsection: {self.text}"
        elif self.content_type == "list_item":
            return f"List item: {self.text}"
        elif self.content_type == "footnote":
            return f"Note: {self.text}"
        else:
            return self.text
 
    def _get_metadata(self) -> dict:
        """
        Generates metadata for the text block.
        """
        metadata = super()._get_metadata()
        metadata.update({
            'font_styles': list(self.font_styles),
            'content_type': self.content_type,
            'block_type': 'text'
        })
        if self.font_size != -1:
            metadata['font_size'] = self.font_size # only supply font size if block has uniform content
        return metadata

    def generate_chunk(self) -> dict:
        """
        Generates a chunk of text with metadata for LlamaIndex embedding.
        """
        return {
            'content': self._get_clean_content(),
            'metadata': self._get_metadata()
        }

    def __str__(self):
        result = f"Text block at: ({self.x0}, {self.y0}) - ({self.x1}, {self.y1})\n"
        result += f"Of size: {self.width} x {self.height} pt\n"
        result += f"Order: {self.page_position[0]} of {self.page_position[1]} on page {self.page}\n"
        result += f"Font size: {self.font_size} pt\n"
        result += f"Font styles: {self.font_styles}\n"
        result += f"Content type: {self.content_type}\n"
        result += f"-------------------Text:-------------------\n{self.text}\n-------------------------------------------\n"
        return result

class ImageBlock(ContentBlock):
    """
    A content block that contains an image and its data. Meant to store structured image content and metadata.
    """
    def __init__(self, pos:tuple,
                 block_index:int,
                 blocks_on_page:int,
                 page:int,
                 caption:str): # caption should be result of BLIP processing
    
        # initialize parent
        super().__init__(pos, block_index, blocks_on_page, page)

        assert isinstance(caption, str), f"Caption must be a string, instead got {type(caption)}."

        # set image attributes
        self.size = self.width * self.height
        self.caption = caption

    def _get_metadata(self) -> dict:
        """
        Generates metadata for the image block.
        """
        metadata = super()._get_metadata()
        metadata.update({
            'image_size': self.size,
            'block_type': 'image'
        })
        return metadata

    def generate_chunk(self) -> dict:
        """
        Generates an image chunk with metadata for LlamaIndex embedding.
        """
        return {
            'content': f"Image description: {self.caption}",
            'metadata': self._get_metadata()
        }

    def __str__(self):
        result = f"Image block at: ({self.x0}, {self.y0}) - ({self.x1}, {self.y1})\n"
        result += f"Of size: {self.width} x {self.height} pt\n"
        result += f"Order: {self.page_position[0]} of {self.page_position[1]} on page {self.page}\n"
        result += f"Captioned: {self.caption}\n"
        return result



