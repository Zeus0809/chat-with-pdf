from llama_cpp import Llama

llm = Llama(
	model_path="./local_models/embed/nomic-embed-text-v2-moe.Q8_0.gguf",
    embedding=True,
    verbose=False,
    n_threads=8
)

embeddings = llm.embed("If the model is specifically an embedding-only model (which appears to be the case for nomic-embed-text-v2-moe), then you should only use the embedding functionality and not try to generate text.If you want both functionalities, you may need separate models:One for embeddings (like nomic-embed-text-v2-moeOne for text generation (like mistral-7b-instruct)For debugging, try this modified version of your script which focuses solely on the embedding capability without trying to generate text.")
print(embeddings)
print(len(embeddings))
