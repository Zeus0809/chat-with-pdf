from llama_index.core.embeddings import MultiModalEmbedding
from llama_cpp import Llama
from typing import Optional, List
from llama_index.core.bridge.pydantic import PrivateAttr

class LlamaCppEmbedding(MultiModalEmbedding):
    """"
    Multi-modal embedding class using llama.cpp for both text and image embeddings. Image model initialization to be added later.
    Uses llama.cpp to run custom embedding models that llama_index doesn't support (GGUF).
    """

    # private attributes that won't be serialized
    _text_model: Llama = PrivateAttr()
    _image_model: Optional[Llama] = PrivateAttr(default=None)

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,
        n_threads: int = 8,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the LlamaCPP embedding model.
    
        Args:
            model_path: Path to the GGUF model file
            n_ctx: Context window size (can be small for embeddings)
            n_threads: Number of CPU threads to use
            verbose: Whether to print verbose output
        """

        # Initialize the llama.cpp model for embeddings
        self._text_model = Llama(
            model_path = model_path,
            embedding = True,
            n_ctx = n_ctx,
            n_threads = n_threads,
            verbose = verbose
        )

        print(f"\n--Constructor executed, _text_model created: {self._text_model}\n")

        super().__init__(**kwargs)

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single chunk of text.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the text embedding vector
        """
        # Call the embed method of our llama.cpp model
        return self._text_model.embed(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple text chunks.

        Args:
            texts: List of chunks to embed

        Returns:
            List of embedding vectors, one for each input chunk
        """
        # Process each chunk separately and return a list of embeddings
        return [self._text_model.embed(text) for text in texts]
        
    # Image embed methods: TBD

    def _get_image_embedding(self, img_file_path: str) -> List[float]:
        """
        Get embedding for a single image (TBD)

        Args:
            img_file_path: Path to the image file
        
        Returns:
            List of floats representing the image embedding vector
        """
        raise NotImplementedError("Image embedding is not implemented yet in LlamaCppEmbedding")
    
    def _get_image_embeddings(self, image_paths: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple images (TBD)

        Args:
            image_paths: List of paths to image files
        
        Returns:
            List of embedding vectors, one for each input image
        """
        raise NotImplementedError("Batch image embedding is not implemented yet in LlamaCppEmbedding")
    
    async def _aget_image_embedding(self, img_file_path: str) -> List[float]:
        """
        TBD
        """
        raise NotImplementedError("Async image embedding is not implemented yet in LlamaCppEmbedding")
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """
        TBD
        """
        raise NotImplementedError("Async query embedding is not implemented yet in LlamaCppEmbedding")

    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for a query string.

        Args:
            query: The query text to embed

        Returns:
            List of floats representing the query embedding vector
        """
        # For text queries, we can use the same embedding approach as normal text
        return self._text_model.embed(query)
    