from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.llms.llama_cpp import LlamaCPP
from src.backend.integrations import LlamaCppEmbedding
from src.backend.hardware import get_optimal_config
from src.backend.llm import ModelManager
import os, time, shutil
from dotenv import load_dotenv

class PDFAgent():

    def __init__(self):
        load_dotenv(verbose=True)
        self.optimal_model_config = get_optimal_config()

        # download embedding and chat models from HF, if not already downloaded
        self.model_manager = ModelManager()

        # Initialize embed and chat models
        Settings.embed_model = LlamaCppEmbedding(model_path=self.model_manager.embed_model_path, verbose=False, **self.optimal_model_config)
        Settings.llm = LlamaCPP(model_path=self.model_manager.chat_model_path, verbose=False, model_kwargs=self.optimal_model_config)

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
        self._query_engine = self._index.as_query_engine(streaming=True)
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
    









