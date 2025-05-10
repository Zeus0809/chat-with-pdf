from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings


# setting our own embedding model
Settings.embed_model = ""

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query(input("Ask me a question about CCCR: "))
print(response)
