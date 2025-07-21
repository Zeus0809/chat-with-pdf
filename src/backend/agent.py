from llama_index.core import VectorStoreIndex, Document, Settings, SimpleDirectoryReader
from llamaindex_utils.integrations import LlamaCppEmbedding, DockerLLM
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv
from typing import List
import os, time, shutil, requests, subprocess, platform

load_dotenv(verbose=True)

CHAT_MODELS = {
        "mistral-7b" : "./local_models/text/mistral-7b-instruct-v0.1.Q5_0.gguf",
        "yi-9b" : "hf.co/bartowski/Yi-1.5-9B-Chat-GGUF:IQ4_NL",
        "gemma3n-docker" : "ai/gemma3n",
        "gemma3n-ollama" : "gemma3n:latest"
        # Try other models during development process -> pick one in the end
    }

def messages_to_prompt(messages: List[dict]) -> str:
    """
    Converts chat messages from LlamaIndex to Mistral Instruct format.
    
    Mistral Instruct format: <s>[INST] user_message [/INST] assistant_response</s>
    Multi-turn: <s>[INST] msg1 [/INST] resp1</s><s>[INST] msg2 [/INST] resp2</s>
    """
    prompt = "<s>"  # Start with opening token
    
    for message in messages:
        if message.role == 'system':
            # System message goes first, no additional <s>
            prompt += f"{message.content}\n"
        elif message.role == 'user':
            prompt += f"[INST] {message.content} [/INST]"
        elif message.role == 'assistant':
            prompt += f" {message.content}</s><s>"
    
    # Clean up trailing <s> if present
    if prompt.endswith("<s>"):
        prompt = prompt[:-3]
    
    return prompt

def completion_to_prompt(completion: str) -> str:
    """
    Fallback formatting function for single completions (rarely used in RAG).
    """
    return f"[INST] {completion} [/INST]"

class PDFAgent():

    def __init__(self, llm_backend: str = "ollama"):

        # Initialize embedding model
        Settings.embed_model = LlamaCppEmbedding(model_path=os.getenv('EMBED_MODEL_PATH'), verbose=False)
        self._embed_model_path = os.getenv('EMBED_MODEL_PATH')

        # Initialize chat model with the specified backend
        if llm_backend == "llamacpp":
            self._chat_model = LlamaCPP(model_path=CHAT_MODELS["mistral-7b"],
                                        temperature=0.1,
                                        max_new_tokens=768,
                                        context_window=4096,
                                        messages_to_prompt=messages_to_prompt,
                                        completion_to_prompt=completion_to_prompt,
                                        verbose=True)
            self._chat_model_path = CHAT_MODELS["mistral-7b"]
            print("\n\n###-Chat model initialized: LlamaCPP with Mistral 7B-###\n\n")
        elif llm_backend == "ollama":
            self.ensure_ollama_running()
            # Initialize chat model with Ollama
            self._chat_model = Ollama(model=CHAT_MODELS["gemma3n-ollama"], temperature=0.1)
            print("\n\n###-Chat model initialized: Ollama with Gemma3n-###\n\n")
        elif llm_backend == "docker":
            self.ensure_docker_running()
            # Initialize chat model with Ollama using Docker Model Runner (experiment)
            self._chat_model = DockerLLM(model=CHAT_MODELS["gemma3n-docker"])
            print("\n\n###-Chat model initialized: Docker Model Runner with Gemma3n-###\n\n")
        else:
            raise ValueError(f"Unsupported LLM backend: {llm_backend}. Available options: 'ollama', 'llamacpp', 'docker'.")

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
                time.sleep(1)
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

    def ensure_ollama_running(self) -> None:
        """
        Ensures that the Ollama service is running. If not, starts it.
        """
        try:
            requests.get("http://localhost:11434/api/tags", timeout=3)
        except:
            # print("--Starting Ollama...")
            if platform.system() == "Darwin":
                subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif platform.system() == "Windows":
                try:
                    subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
                except Exception as e:
                    print("--Failed to start ollama on Windows--")
                    # TODO: replace with actual UI error
            else:
                print("--Unsupported OS. Failed to start the ollama service.")
                return
            # Wait for Ollama to start with timeout
            for _ in range(10):  # Try for up to 10 seconds
                time.sleep(1)
                try:
                    requests.get("http://localhost:11434/api/tags", timeout=2)
                    print("--Ollama ready--")
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
        self._query_engine = self._index.as_query_engine(llm=self._chat_model)
        print(f"--Index created in {round(time.time() - start, 2)}s.--")

    def create_index_from_chunks(self, chunks: List[dict]) -> None:
        """
        Creates a vector store index from content chunks with rich metadata.
        Each chunk becomes a separate Document for better retrieval granularity.
        """
        assert isinstance(chunks, list), f"Chunks should be a list, instead got {type(chunks)}."

        start = time.time()

        # Create LlamaIndex documents from prepared chunks
        documents = []
        for chunk in chunks:
            assert isinstance(chunk, dict), f"Each chunk should be a dictionary, instead got {type(chunk)}."
            doc = Document(
                text=chunk['content'],
                metadata=chunk['metadata']
            )
            documents.append(doc)

        self._index = VectorStoreIndex.from_documents(documents, show_progress=True)
        assert self._index is not None, "Index is None. Create an index before creating a query engine."
        self._query_engine = self._index.as_query_engine(llm=self._chat_model)
        print(f"--Index created in {round(time.time() - start, 2)} seconds--")

    def ask_agent(self, prompt: str) -> str:
        """
        Asks the agent a question from the user and returns the response.
        """
        assert isinstance(prompt, str), f"Prompt should be a stirng, instead got {type(prompt)}."
        assert self._query_engine is not None, "Query engine is None. Please call PDFAgent.create_index_from_chunks() before asking the agent."
        start = time.time()
        response = self._query_engine.query(prompt)
        assert response, "Response from the agent is None."
        print(f"--Agent response generated in {round(time.time() - start, 2)}s.--\n")
        print(f"Agent response: {response}")
        return response



