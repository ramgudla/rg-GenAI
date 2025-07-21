# https://luca-bindi.medium.com/oracle23ai-simplifies-rag-implementation-for-enterprise-llm-interaction-in-enterprise-solutions-d865dacdd1ed

import sys
import os

from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

n = len(sys.argv)
if n == 1:
    vector_store = "chroma"
else:
    vector_store = sys.argv[1]

current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from utils.genai_wrapper import GenAIWrapper

## Get the OCI LLM and Embedding models
genai_wrapper = GenAIWrapper()

llm = genai_wrapper.llm

user_question = ("What are the highlights of budget?")
print("\nThe prompt to the LLM will be:", user_question)
# Set up the template for the questions and context, and instantiate the database retriever object
template = """Answer the question based only on the following context:
            {context} Question: {user_question}"""
prompt = PromptTemplate.from_template(template)

match vector_store:
    case "ora23ai":
        # Optional. Just get relevant docs from oracle23ai vector store
        oracle_vs = genai_wrapper.get_ora23ai_vs()
        query = "What are the highlights of budget?"
        docs = oracle_vs.similarity_search(query)
        print(docs)

        chain = (
            {"context":  oracle_vs.as_retriever(), "user_question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = chain.invoke(user_question)
        print(user_question)
        print(response)
        
    case "chroma":
        # Optional. Just get relevant docs from local chrome db
        chroma_vs = genai_wrapper.get_chroma_vs()
        query = "What are the highlights of budget?"
        docs = chroma_vs.similarity_search(query)
        print(docs)

        chain = (
            {"context": chroma_vs.as_retriever(), "user_question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = chain.invoke(user_question)
        print(user_question)
        print(response)

    case "faiss":
        # Optional. Just get relevant docs from local chrome db
        faiss_vs = genai_wrapper.get_faiss_vs()
        query = "What are the highlights of budget?"
        docs = faiss_vs.similarity_search(query)
        print(docs)

        chain = (
            {"context": faiss_vs.as_retriever(search_type="similarity", search_kwargs={"k": 5}), "user_question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = chain.invoke(user_question)
        print(user_question)
        print(response)

print('\nSuccessfully retrieved content from llm using the context retrieved from the vector store.')