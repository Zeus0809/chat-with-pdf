from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow.handler import WorkflowHandler
from llama_index.core.tools import FunctionTool

from llamaindex_utils.integrations import LlamaCppEmbedding, DockerLLM

import os, time, shutil, requests, subprocess, platform, json
from dotenv import load_dotenv

load_dotenv(verbose=True)

# Parse CHAT_MODELS from JSON environment variable
CHAT_MODELS = json.loads(os.getenv('CHAT_MODELS', '{}'))

# Load environment variables for agent configuration
AGENT_SYS_PROMPT = os.getenv('AGENT_SYS_PROMPT')
RAG_TOOL_NAME = os.getenv('RAG_TOOL_NAME', 'rag_query')
RAG_TOOL_DESC = os.getenv('RAG_TOOL_DESC')

class PDFAgent():

    def __init__(self, llm_backend: str = "docker"):

        # Initialize embedding model
        Settings.embed_model = LlamaCppEmbedding(model_path=os.getenv('EMBED_MODEL_PATH'), verbose=False)
        self._embed_model_path = os.getenv('EMBED_MODEL_PATH')

        # Initialize chat model with the specified backend
        if llm_backend == "docker":
            self._ensure_docker_running()
            # Initialize chat model with Ollama using Docker Model Runner (experiment)
            self._chat_model = DockerLLM(model=CHAT_MODELS["gemma3n"])
            print("\n\n###-Chat model initialized: Docker Model Runner with Gemma3n-###\n\n")
        else:
            raise ValueError(f"Unsupported LLM backend: {llm_backend}. Available options: 'docker'.")

        # Index and query engine
        self._index = None
        self._query_engine = None

        # Agent with function caling
        self._function_agent = None

    def _ensure_docker_running(self) -> None:
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

    def _initialize_agent(self) -> None:
        """Create tools from existing functionality and pass them to FunctionAgent"""
        assert isinstance(self._query_engine, BaseQueryEngine), f"Make sure _query_engine is created before you initialize the agent. Type received: {type(self._query_engine)}"
        assert isinstance(self._chat_model, DockerLLM), f"Make sure _chat_model is initialized before initializing the agent. Type received: {type(self._chat_model)}" 
        
        # RAG tool with debug wrapper
        rag_tool = FunctionTool.from_defaults(
            fn=self._rag_query,
            name=RAG_TOOL_NAME,
            description=RAG_TOOL_DESC
        )
        
        # page nav tool
        # goto_page_tool = FunctionTool.from_defaults(
        #     fn=self.go_to_page,
        #     name="goto_page",
        #     description="Use this tool when the user asks to be taken to a particular page in the PDF document. Input: page number. You should obtain the page number either from the user question, or from the conversation context."
        # )
        
        # agent with more explicit prompting
        self._function_agent = FunctionAgent(
            tools=[rag_tool],
            llm=self._chat_model,
            system_prompt=AGENT_SYS_PROMPT
        )

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
        self._initialize_agent()
        print(f"--Function Agent initialized--")

    def ask_agent(self, prompt: str) -> WorkflowHandler:
        """
        Asks the agent a question from the user and returns the WorkflowHandler for streaming.
        """
        assert isinstance(prompt, str), f"Prompt should be a string, instead got {type(prompt)}."
        
        # Run the agent and return the handler (NOT awaited)
        handler = self._function_agent.run(user_msg=prompt)
        
        return handler
    
    def _go_to_page():
        pass

    # Tool: Create a wrapper for the query engine
    def _rag_query(self, query: str) -> str:
        print(f"🔧 RAG TOOL CALLED with query: {query}")
        result = self._query_engine.query(query)
        print(f"🔧 RAG TOOL RESULT: {str(result)[:20]}...")
        return str(result)
    













