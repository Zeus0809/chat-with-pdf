from llama_index.core.base.llms.types import ( 
    LLMMetadata,
    CompletionResponseGen,
    CompletionResponse,
    ChatMessage,
    ChatResponse,
    ChatResponseGen
)
from llama_index.core.llms.callbacks import llm_completion_callback, llm_chat_callback
from llama_index.core.bridge.pydantic import PrivateAttr, Field
from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.core.llms.function_calling import FunctionCallingLLM
from typing import Optional, List, Any
from llama_cpp import Llama
import requests, json

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
        
        super().__init__(**kwargs)

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

class DockerLLM(FunctionCallingLLM):
    """
    Custom LLM class to use Docker Model Runner for chat models inside LlamaIndex's RAG pipeline.
    Supports function calling.
    Docker Model Runner API docs: https://docs.docker.com/ai/model-runner/
    
    Main endpoints:
        /engine/llama.cpp/v1/models
        /engine/llama.cpp/v1/completions
        /engine/llama.cpp/v1/chat/completions
    """

    model: str = Field(
        description="Docker Model Runner model name.",
        examples=["ai/qwen3", "ai/gemma3n"],
        min_length=1
    )
    base_url: str = Field(
        default="http://localhost:12434",
        description="Docker Model Runner API base URL."
    )
    max_tokens: int = Field(
        default=512,
        description="Maximum number of tokens to generate in completion.",
        ge=1,
        le=4096
    )
    temperature: float = Field(
        default=1.0,
        description="Temperature for sampling during text generation.",
        ge=0.0,
        le=2.0  # OpenAI-style range
    )
    timeout: float = Field(
        default=60.0,
        description="Timeout for HTTP requests to the Docker Model Runner."
    )

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:12434",
        temperature: float = 0.5,
        timeout: float = 60.0,
        max_tokens: int = 512,
        *args: Any,
        **kwargs: Any
    ) -> None:
        # Pass all fields to parent so Pydantic can create them
        super().__init__(
            model=model,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
            *args, **kwargs
        )

    @classmethod
    def class_name(cls) -> str:
        return "docker_llm"
    
    @property
    def metadata(self) -> LLMMetadata:
        """Docker LLM metadata."""
        return LLMMetadata(
            is_chat_model=True,
            model_name=self.model
        )
    
    def _get_completions_endpoint(self) -> str:
        return f"{self.base_url}/engines/llama.cpp/v1/completions"
    
    def _get_chat_endpoint(self) -> str:
        return f"{self.base_url}/engines/llama.cpp/v1/chat/completions"
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        Implementing the _complete method as instructed by CustomLLM.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            **kwargs
        }
        response = requests.post(url=self._get_completions_endpoint(), json=payload, timeout=self.timeout)
        response.raise_for_status()
        response_data = response.json()
        return CompletionResponse(
            text=response_data["choices"][0]["text"]
        )
    
    @llm_completion_callback()
    def acomplete(self):
        """Just a placeholder"""
        raise NotImplementedError('acomplete is not yet implemented in DockerLLM.')
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """
        Implementing the _stream_complete method as instructed by CustomLLM.
        The method should return a generator function, so we can pull tokens from it on the outside.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            **kwargs
        }

        def gen() -> CompletionResponseGen:
            response = requests.post(
                url=self._get_completions_endpoint(),
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            text = ""
            # The http response needs to be iterated over line by line as it comes in
            for line in response.iter_lines(decode_unicode=True):
                if not line or line == "[DONE]":
                    continue

                # Handle Server-Sent Events (SSE) format (remove 'data' prefix)
                if line.startswith("data: "):
                    line = line[6:]

                try:
                    data = json.loads(line)

                    delta = ""
                    # flexible delta extraction depending on response format (in case DMR makes changes in the future)
                    if "choices" in data and len(data["choices"]) > 0:
                        choice = data["choices"][0]
                        delta = (choice.get("text", "") or
                                 choice.get("delta", {}).get("content", "") or
                                 choice.get("delta", {}).get("text", ""))
                        
                    if delta:
                        text += delta
                        yield CompletionResponse(delta=delta, text=text, raw=data)
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue

        return gen()

    @llm_completion_callback()
    def astream_complete(self):
        """Just a placeholder"""
        raise NotImplementedError('astream_complete is not yet implemented in DockerLLM.')

    def _prepare_chat_with_tools(self):
        """Just a placeholder"""
        raise NotImplementedError('_prepare_chat_with_tools is not yet implemented in DockerLLM.')
    
    @llm_chat_callback()
    def chat(self):
        """Just a placeholder"""
        raise NotImplementedError('chat is not yet implemented in DockerLLM.')
    
    @llm_chat_callback()
    def achat(self):
        """Just a placeholder"""
        raise NotImplementedError('achat is not yet implemented in DockerLLM.')

    @llm_chat_callback()
    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        """
        Implementing streaming chat with conversation context using Docker Model Runner's chat completions endpoint.
        This method maintains conversation history and returns a generator for streaming responses.
        """
        # Convert LlamaIndex ChatMessage objects to OpenAI-compatible format
        formatted_messages = []
        for message in messages:
            formatted_messages.append({
                "role": message.role.value,  # Convert MessageRole enum to string
                "content": message.content
            })
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            **kwargs
        }

        def gen() -> ChatResponseGen:
            response = requests.post(
                url=self._get_chat_endpoint(),
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            content = ""
            # The http response needs to be iterated over line by line as it comes in
            for line in response.iter_lines(decode_unicode=True):
                if not line or line == "[DONE]":
                    continue

                # Handle Server-Sent Events (SSE) format (remove 'data' prefix)
                if line.startswith("data: "):
                    line = line[6:]

                try:
                    data = json.loads(line)

                    delta = ""
                    # Extract delta from chat completion response format
                    if "choices" in data and len(data["choices"]) > 0:
                        choice = data["choices"][0]
                        # For chat completions, delta is usually in choice.delta.content
                        if "delta" in choice and "content" in choice["delta"]:
                            delta = choice["delta"]["content"]
                        # Fallback for other possible formats
                        elif "message" in choice and "content" in choice["message"]:
                            delta = choice["message"]["content"]
                        elif "text" in choice:
                            delta = choice["text"]
                        
                    if delta:
                        content += delta
                        # Create ChatResponse instead of CompletionResponse for chat methods
                        chat_response = ChatResponse(
                            message=ChatMessage(role="assistant", content=content),
                            delta=delta,
                            raw=data
                        )
                        yield chat_response
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue

        return gen()

    @llm_chat_callback()
    def astream_chat(self):
        """Just a placeholder"""
        raise NotImplementedError('astream_chat is not yet implemented in DockerLLM.')

    


        