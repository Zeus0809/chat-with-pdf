from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.core.base.response.schema import StreamingResponse

from llamaindex_utils.integrations import LlamaCppEmbedding, DockerLLM

import os, time, shutil, requests, subprocess, platform
from dotenv import load_dotenv

load_dotenv(verbose=True)

CHAT_MODELS = {
        "gemma3n" : "ai/gemma3n",
        "qwen3" : "ai/qwen3",
        "smollm3" : "ai/smollm3",
        "deepseek-llama" : "ai/deepseek-r1-distill-llama"
        # Try other models during development process -> pick one in the end
    }

class PDFAgent():

    def __init__(self, llm_backend: str = "docker"):

        # Initialize embedding model
        Settings.embed_model = LlamaCppEmbedding(model_path=os.getenv('EMBED_MODEL_PATH'), verbose=False)
        self._embed_model_path = os.getenv('EMBED_MODEL_PATH')

        # Initialize chat model with the specified backend
        if llm_backend == "docker":
            self.ensure_docker_running()
            # Initialize chat model with Ollama using Docker Model Runner (experiment)
            self._chat_model = DockerLLM(model=CHAT_MODELS["gemma3n"])
            print("\n\n###-Chat model initialized: Docker Model Runner with Gemma3n-###\n\n")
        else:
            raise ValueError(f"Unsupported LLM backend: {llm_backend}. Available options: 'docker'.")

        # Index and query engine
        self._index = None
        self._query_engine = None

    def ensure_docker_running(self) -> None:
        """
        Ensures that the Docker engine is running so that Docker Model Runner is available. If not, starts it.
        """
        try:
            requests.get("http://localhost:12434/engine/llama.cpp/v1/models", timeout=3)
        except:
            # starting docker
            if platform.system() == "Darwin": # macOS
                subprocess.Popen(['open', '-a', 'Docker'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(4)
                try:
                    subprocess.run(['osascript', '-e', 'tell application "System Events" to set visible of process "Docker Desktop" to false'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except:
                    print("--Failed to hide Docker window--")
            elif platform.system() == "Windows":
                try:
                    subprocess.Popen(['cmd', '/c', 'start', 'Docker Desktop'])
                except Exception as e:
                    print("--Failed to start docker on Windows--")
                    # TODO: replace with actual UI error
            else:
                print("--Unsupported OS. Failed to start the docker engine.")
                return
            # Wait for Docker to start with timeout
            for _ in range(8):  # Try for up to 8 seconds
                time.sleep(1)
                try:
                    requests.get("http://localhost:12434/engine/llama.cpp/v1/models", timeout=2)
                    print("--Docker Model Runner ready--")
                    break
                except:
                    continue

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
        assert self._query_engine is not None, "Query engine is None. Please call PDFAgent.create_index_from_chunks() before asking the agent."
        start = time.time()
        response = self._query_engine.query(prompt) # returns a generator
        assert response, "Response from the agent is None."
        print(f"--Agent response generator ready in {round(time.time() - start, 2)}s.--\n")
        return response
    









