import sys
import os
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI

#In this demo we will explore using Streamlit session to store chat messages

#Step 1 - setup OCI Generative AI llm

current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from util.genai_wrapper import GenAIWrapper

## Get the OCI LLM and Embedding models
genai_wrapper = GenAIWrapper()

llm = genai_wrapper.llm
embeddings = genai_wrapper.embeddings

#Step 2 - here we create a history with a key "chat_messages.

#StreamlitChatMessageHistory will store messages in Streamlit session state at the specified key=.
#A given StreamlitChatMessageHistory will NOT be persisted or shared across user sessions.

history = StreamlitChatMessageHistory(key="chat_messages")

#Step 3 - here we create a memory object

memory = ConversationBufferMemory(chat_memory=history)

#Step 4 - here we create template and prompt to accept a question

template = """You are an AI chatbot having a conversation with a human.
Human: {human_input}
AI: """
prompt = PromptTemplate(input_variables=["human_input"], template=template)

#Step 5 - here we create a chain object

llm_chain = LLMChain(llm=genai_wrapper.llm, prompt=prompt, memory=memory)

#Step 6 - here we use streamlit to print all messages in the memory, create text imput, run chain and
#the question and response is automatically put in the StreamlitChatMessageHistory

import streamlit as st

st.title('🦜🔗 Welcome to the ChatBot')
for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

if x := st.chat_input():
    st.chat_message("human").write(x)

    # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
    response = llm_chain.invoke(x)
    st.chat_message("ai").write(response["text"])