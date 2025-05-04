from llama_cpp import Llama
import pprint

my_local_llm = Llama(model_path="./local_models/text/Yi-1.5-9B-Chat-IQ4_NL.gguf",
                     chat_format="chatml",
                     n_ctx=4096,
                     n_threads=8,
                     use_mlock=False,
                     verbose=False)

go_on = "y"
while go_on == "y":

    question = input("Ask me a question: ")
    messages = [ {"role": "system", "content": "You are a helpful multilingual assistant"}, {"role": "user", "content": question} ]
    response = my_local_llm.create_chat_completion(messages, max_tokens=200)
    print(response["choices"][0]["message"]["content"])
    go_on = input("\n\nGo on? y/n")

