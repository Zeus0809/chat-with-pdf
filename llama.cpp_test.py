from llama_cpp import Llama
import pprint

my_local_llm = Llama(model_path="./local_models/text/Yi-1.5-9B-Chat-IQ4_NL.gguf", chat_format="llama-2")

messages = [ {"role": "system", "content": "You are a helpful multilingual assistant."}, {"role": "user", "content": "Hello Yi."} ]

response = my_local_llm.create_chat_completion_openai_v1(messages)

print(response.choices[0].message.content)
