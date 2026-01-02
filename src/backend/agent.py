from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.core.base.response.schema import StreamingResponse

from src.backend.integrations import LlamaCppEmbedding

import os, time, shutil
from dotenv import load_dotenv

load_dotenv(verbose=True)

CHAT_MODELS = {
        "gemma3n" : "ai/gemma3n"
        # Try other models during development process -> pick one in the end
    }

class PDFAgent():

    def __init__(self, llm_backend: str = "docker"):

        # Initialize embedding model
        Settings.embed_model = LlamaCppEmbedding(model_path=os.getenv('EMBED_MODEL_PATH'), verbose=False)
        self._embed_model_path = os.getenv('EMBED_MODEL_PATH')
        self._chat_model = None

        # Index and query engine
        self._index = None
        self._query_engine = None

    def create_index(self, file_path: str) -> None:
        """
        The simplest, baseline way to create an index using LlamaIndex.
        """
        # copy file into ~/storage/data to only index the file we need
        shutil.copy(file_path, os.getenv('DATA_PATH'))
        start = time.time()
        documents = SimpleDirectoryReader(os.getenv('DATA_PATH')).load_data()
        self._index = VectorStoreIndex.from_documents(documents, show_progress=True)
        assert self._index is not None, "Index is None. Create an index before creating a query engine."
        self._query_engine = self._index.as_query_engine(llm=self._chat_model, streaming=True)
        print(f"--Index created in {round(time.time() - start, 2)}s.--")

    def ask_agent(self, prompt: str) -> StreamingResponse:
        """
        Asks the agent a question from the user and returns the response.
        """
        assert isinstance(prompt, str), f"Prompt should be a string, instead got {type(prompt)}."
        assert self._query_engine is not None, "Query engine is None. Please call PDFAgent.create_index() before asking the agent."
        start = time.time()
        response = self._query_engine.query(prompt) # returns a generator
        assert response, "Response from the agent is None."
        print(f"--Agent response generator ready in {round(time.time() - start, 2)}s.--\n")
        return response
    









