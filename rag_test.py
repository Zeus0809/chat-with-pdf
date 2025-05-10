from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_cpp_embedding import LlamaCppEmbedding

# setting our own embedding model
Settings.embed_model = LlamaCppEmbedding(model_path="./local_models/embed/nomic-embed-text-v2-moe.Q8_0.gguf")

print("\nModel name: ", Settings.embed_model.model_name, "\n")

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query(input("Ask me a question about CCCR: "))
print(response)
