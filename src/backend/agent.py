from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llamaindex_utils.llama_cpp_embedding import LlamaCppEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
import os
import time

EMBED_MODEL_PATH = "./local_models/embed/nomic-embed-text-v2-moe.Q8_0.gguf"
CHAT_MODELS = {
        "mistral-7b" : "./local_models/text/mistral-7b-instruct-v0.1.Q5_0.gguf",
        # Try other models during development process -> pick one in the end
    }

class PDFAgent():

    def __init__(self):
        # Initialize embedding model
        Settings.embed_model = LlamaCppEmbedding(model_path=CHAT_MODELS["mistral-7b"], verbose=False)
        self._embed_model_path =  EMBED_MODEL_PATH
        # Initialize chat model
        self._chat_model = LlamaCPP(model_path=CHAT_MODELS["mistral-7b"], verbose=False)
        self._chat_model_path = CHAT_MODELS["mistral-7b"]
        # Index and query engine
        self._index = None
        self._query_engine = None

    @property
    def embed_model_name(self) -> str:
        model_name = os.path.basename(self._embed_model_path)
        assert isinstance(Settings.embed_model, LlamaCppEmbedding), f"Embedding model ({model_name}) failed to initialize."
        return model_name

    @property
    def chat_model_name(self) -> str:
        assert self._chat_model, "Chat model failed to initialize."
        return os.path.basename(self._chat_model_path)

    @property
    def get_index_summary(self) -> str:
        return self._index.summary

    def create_index_from_markdown(self, markdown: str, metadata: dict=None) -> None:
        """
        Creates a vector store index from the provided markdown content and supplies a query engine.
        """
        assert markdown, "Markdown content cannot be empty."
        start = time.time()
        document = Document(text=markdown, metadata=metadata or {})
        self._index = VectorStoreIndex.from_documents([document])
        assert self._index is not None, "Index is None. Create an index before creating a query engine."
        self._query_engine = self._index.as_query_engine(llm=self._chat_model)
        print(f"--Index created in {round(time.time() - start, 2)} seconds--")

    def ask_agent(self, prompt: str) -> str:
        """
        Asks the agent q question from the user and returns the response.
        """
        assert prompt, "Prompt cannot be empty."
        assert self._query_engine is not None, "Query engine is None. Create an index from your document before asking the agent."
        start = time.time()
        response = self._query_engine.query(prompt)
        print(f"--Agent response generated in {round(time.time() - start, 2)} seconds--")
        print(f"Response type: {type(response)}, \nResponse itself: {response}")



    
