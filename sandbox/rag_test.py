from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llamaindex_utils.llama_cpp_embedding import LlamaCppEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
import time

# setting our own embedding model
Settings.embed_model = LlamaCppEmbedding(model_path="./local_models/embed/nomic-embed-text-v2-moe.Q8_0.gguf")

# Configuring and setting or chat model
chat_model = LlamaCPP(model_path = "./local_models/text/mistral-7b-instruct-v0.1.Q5_0.gguf")

start = time.time()
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
end = time.time()

# print("Vector Store: ", index.vector_store)
# print("Storage Context: ", index.storage_context)
# print("Summary: ", index.summary)

# print("Index created in: ", end - start)

# start = time.time()
# query_engine = index.as_query_engine(chat_model)
# response = query_engine.query(input("Ask me a question about CCCR: "))
# print(response)
# end = time.time()

# print("Response generated in: ", end - start)
