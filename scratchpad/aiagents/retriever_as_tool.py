import sys
import os

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore
from langchain.agents import Tool
from langchain.document_loaders import TextLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.pydantic_v1 import BaseModel, Field
from langchain.agents import create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel, Field
from langchain.agents import tool
import numpy as np
import langchain

langchain.__version__  # Should be 0.1.0 , latest version giving errors while using OAI tools agent

# import openai

# openai.__version__

## Setting up LLM

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_DEPLOYMENT_NAME =  "gpt-35-turbo-16k" #"gpt4-turbo"
# OPENAI_DEPLOYMENT_ENDPOINT = "https://<???>.openai.azure.com/"
# OPENAI_DEPLOYMENT_VERSION = "2023-12-01-preview"
# OPENAI_MODEL_NAME = "gpt-35-turbo-16k" #"gpt-4" 

# OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada"
# OPENAI_ADA_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"

# llm = AzureChatOpenAI(
#     deployment_name=OPENAI_DEPLOYMENT_NAME,
#     model_name=OPENAI_MODEL_NAME,
#     openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
#     openai_api_version=OPENAI_DEPLOYMENT_VERSION,
#     openai_api_key=OPENAI_API_KEY,
#     openai_api_type="azure",
#     temperature=0.1,
# )

# embeddings = OpenAIEmbeddings(
#     deployment=OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
#     model=OPENAI_ADA_EMBEDDING_MODEL_NAME,
#     openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
#     openai_api_type="azure",
#     chunk_size=1,
#     openai_api_key=OPENAI_API_KEY,
#     openai_api_version=OPENAI_DEPLOYMENT_VERSION,
# )

current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from utils.genai_wrapper import GenAIWrapper

## Get the OCI LLM and Embedding models
genai_wrapper = GenAIWrapper()

llm = genai_wrapper.llm
embeddings = genai_wrapper.embeddings

# langchain properties for hub.pull()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "xxxxx"
os.environ["LANGCHAIN_PROJECT"] = 'default'

#Loading documents
loader = TextLoader("data/globalcorp_hr_policy.txt")
documents = loader.load()
collection_name = "hrpolicy"

## Creating Retriever

# This text splitter is used to create the child documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    persist_directory="chroma_index",
    collection_name=collection_name,
    embedding_function=embeddings,
)

# The storage layer for the parent documents
local_store = LocalFileStore("chroma_index")
store = create_kv_docstore(local_store)
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# run only once
# vectorstore.persist()
# retriever.add_documents(documents, ids=None)

## QnA System

#correct responses
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)
response = qa({"query": "What is the allocated budget for communication initiatives?"})
print(response)

qa({"query": "How many maternity leaves are offered?"})
qa({"query": "What is the probationary period?"})
qa({"query": "Is the work hours in Germany different than United States?"})
qa({"query": "What is the probationary period for new employees in USA?"})
qa(
    {
        "query": "What is the difference in the number of work hours in Germany vs. United States?"
    }
)
qa({"query": "Can I reimburse travel expenses?"})
qa(
    {
        "query": "I started with the company on 1 December 2023.\
        Is my probationary period over if the date today is 26 Jan 2024?"
    }
)

## incorrect responses

# incorrect as the conversion used is wrong. We need to fix this!
qa(
    {
        "query": "What is the percentage difference in the annual budget for Japan and US?"
    }
)

# Results are still slightly off
qa(
    {
        "query": "What is the percentage difference in the annual budget for Japan and US if 1 USD = 147.72 JPY?"
    }
)

# incorrect as technically US has higher budget after conversion
qa({"query": "Which country has the highest budget?"})

## Setting up ReAct Agent

#Common Tools

# convert PDR retriever into a tool
tool_search = create_retriever_tool(
    retriever,
    "search_hr_policy",
    "Searches and returns excerpts from the HR policy.",
)

# under the hood it will call the get_relevant_documents() function and return the list of parent chunks
tool_search.func

# useful to check the schema to verify the expected parameters for the function
tool_search.args_schema.schema()

def value_to_float(x):
    if int(x):
        return int(x)
    if type(x) == float or type(x) == int:
        return x
    x = x.upper()
    if "MILLION" in x:
        if len(x) > 1:
            return float(x.replace("MILLION", "")) * 1000000
        return 1000000.0
    if "BILLION" in x:
        return float(x.replace("BILLION", "")) * 1000000000

    return 0.0


def convert_currency_to_usd(amount: str) -> int:
    "Converts currency into USD"

    if "¥" in amount:
        exclude_symbol = amount.replace("¥", "")
        amount_in_numbers = value_to_float(exclude_symbol)
        return amount_in_numbers / 147.72
    if "$" in amount:
        exclude_symbol = amount.replace("$", "")
        return value_to_float(exclude_symbol)
    if "JPY" in amount:
        exclude_symbol = amount.replace("JPY", "")
        return int(exclude_symbol) / 147.72
    if "USD" in amount:
        return amount

# It is okay to define single-input tools in this manner.
currency_conversion = Tool(
    name="Currency_conversion",
    func=convert_currency_to_usd,
    description="useful for converting currency into USD. Input should be an amount.",
)

# for multi-input tool, useful to define schema class
class Metrics(BaseModel):
    num1: float = Field(description="Value 1")
    num2: float = Field(description="Value 2")

@tool(args_schema=Metrics)
def perc_diff(num1: float, num2: float) -> float:
    """Calculates the percentage difference between two numbers"""
    return (np.abs(num1 - num2) / ((num1 + num2) / 2)) * 100

## Common Agent prompts

prompt_react = hub.pull("hwchase17/react")
print(prompt_react.template)

## Running the ReAct agent

# List of tools to be used
tools = [tool_search]

# only creates the logical steps for us
react_agent = create_react_agent(llm, tools=tools, prompt=prompt_react)

# executes the logical steps we created
react_agent_executor = AgentExecutor(
    agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True
)

query = "As per the HR policy, Which country has the highest budget?"
react_agent_executor.invoke({"input": query})

query = "Which of the two countries has the highest budget - Japan or Unites States?"
react_agent_executor.invoke({"input": query})

query = "How much is the budget for Japan different than United States?"
react_agent_executor.invoke({"input": query})

# List of tools to be used
tools = [tool_search, currency_conversion]

react_agent = create_react_agent(llm, tools=tools, prompt=prompt_react)
react_agent_executor = AgentExecutor(
    agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True
)

query = "Is the budget for Japan different than United States?"
react_agent_executor.invoke({"input": query})

# Gives close enough response but can be improved with a calculator
query = "Calculate the difference in company budget for Japan and United States?"
react_agent_executor.invoke({"input": query})

## Running the OpenAI Tools Agent
## P.S. This point onwards, I switched to gpt4 for better results

# defining prompt

# system_message = """
# You are very helpful and try your best to answer the questions.
# """

# prompt_oai_tools_simple = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_message),
#         ("user", "{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ]
# )
# tools = [tool_search, currency_conversion, perc_diff]
# prompt_oai_tools = hub.pull("hwchase17/openai-tools-agent")
# oaitools_agent = create_openai_tools_agent(llm, tools, prompt_oai_tools)
# oaitools_agent_executor = AgentExecutor(
#     agent=oaitools_agent, tools=tools, verbose=True, handle_parsing_errors=True
# )

# query = "As per the HR policy, compare the budgets for Japan and US."
# oaitools_agent_executor.invoke({"input": query})