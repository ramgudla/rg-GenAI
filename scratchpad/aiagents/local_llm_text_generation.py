# https://awinml.github.io/llm-ggml-python/

import os
import urllib.request
from llama_cpp import Llama
from langchain_community.llms import LlamaCpp
from langchain_community.llms import Ollama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

def download_file(file_link, filename):
    # Checks if the file already exists before downloading
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(file_link, filename)
        print("File downloaded successfully.")
    else:
        print("File already exists.")

# Dowloading GGML model from HuggingFace
ggml_model_path = "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_0.gguf"
filename = "zephyr-7b-beta.Q4_0.gguf"
download_file(ggml_model_path, filename)

#1:
llamacpp_llm = Llama(model_path="zephyr-7b-beta.Q4_0.gguf", n_ctx=512, n_batch=126)

def generate_text(
    prompt="Who is the CEO of Apple?",
    max_tokens=256,
    temperature=0.1,
    top_p=0.5,
    echo=False,
    stop=["#"],
):
    output = llamacpp_llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=echo,
        stop=stop,
    )
    output_text = output["choices"][0]["text"].strip()
    return output_text


def generate_prompt_from_template(input):
    chat_prompt_template = f"""<|im_start|>system
You are a helpful chatbot.<|im_end|>
<|im_start|>user
{input}<|im_end|>"""
    return chat_prompt_template

question="Who is the CEO of Oracle?"
prompt = generate_prompt_from_template(question)
answer = generate_text(
    prompt,
    max_tokens=356,
)

print("\n Question: " + question)
print("\n Answer: " + answer)

question="Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
prompt = generate_prompt_from_template(question)
answer = generate_text(
    prompt,
    max_tokens=356,
)

print("\n Question: " + question)
print("\n Answer: " + answer)

#2:
#LOAD THE MODEL using langchain llamacpp
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
langchain_llm = LlamaCpp(
   model_path="zephyr-7b-beta.Q4_0.gguf",
   temperature=0.75,
   max_tokens=2000,
   top_p=1,
   callback_manager=callback_manager,
   verbose=True,  # Verbose is required to pass to the callback manager
)

def generate_response_using_langchain(question):
  langchain_llm.invoke(question)

question = """
  Question: A rap battle between Stephen Colbert and John Oliver
  """

print("\n\nQuestion:" + question)
generate_response_using_langchain(question)

#3:
# create a local model from zephyr-7b-beta.Q4_0.gguf
# RUN: ollama create zephyr-7b -f Modelfile

#LOAD THE MODEL using langchain ollama
ollama_llm = Ollama(model="zephyr-7b")
prompt = "Who is the CEO of Oracle?"
answer = ollama_llm.invoke(prompt)

print("\n Question: " + prompt)
print("\n Answer: " + answer)