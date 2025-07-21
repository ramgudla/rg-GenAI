# https://luca-bindi.medium.com/oracle23ai-simplifies-rag-implementation-for-enterprise-llm-interaction-in-enterprise-solutions-d865dacdd1ed

import sys
import os

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

match vector_store:
    case "ora23ai":
        genai_wrapper.persist_ora23ai_vs()
    case "chroma":
        genai_wrapper.persist_chroma_vs()
    case "faiss":
        genai_wrapper.persist_faiss_vs()

print('\nSuccessfully embedded documents into vector store.')