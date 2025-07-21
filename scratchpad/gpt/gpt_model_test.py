# https://awinml.github.io/llm-ggml-python/

import os
import urllib.request
from llama_cpp import Llama
from langchain_community.llms import LlamaCpp
from langchain_community.llms import Ollama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

#1:
llm = Llama(model_path="mymodel/rgAI.gguf", n_ctx=512, n_batch=126)

def generate_text(
    prompt="The quick brown",
    max_tokens=256,
    temperature=0.1,
    top_p=0.5,
    echo=False,
    stop=["#"],
):
    output = llm(
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


