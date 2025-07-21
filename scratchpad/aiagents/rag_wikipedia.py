# # Import the langchain libraries

# %%
import sys
import os

from langchain_community.document_loaders import WikipediaLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# #### Get the OCI LLM and Embedding model

# %%
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from utils.genai_wrapper import GenAIWrapper

## Get the OCI LLM and Embedding models
genai_wrapper = GenAIWrapper()

llm = genai_wrapper.llm
embeddings = genai_wrapper.embeddings



# ### Document Loader
# ###### [Documentation Link](https://python.langchain.com/docs/modules/data_connection/document_loaders/)
#

# %%
docs = WikipediaLoader(query = "Oracle Corporation").load()
print(docs[0].metadata)  # meta-information of the Document
docs[0].page_content[:300]  # a content of the Document



# ##### Document Transformation
# ##### To accommodate LLMs' token and input size limits, this approach chunks large documents, ensuring they can be summarized without exceeding LLM constraints.
# ###### [Link](https://python.langchain.com/docs/modules/data_connection/document_transformers/)

# %%
splitted_docs = RecursiveCharacterTextSplitter(chunk_size = 900, chunk_overlap = 20, length_function=len)

chunks = splitted_docs.split_documents(docs)

# %%
print(f"Total Chunks created {len(chunks)}")
for i, _ in enumerate(chunks):
    print(f"chunk# {i}, size: {chunks[i]}")


# ### Store the embedding data in a vector store (Chroma DB)
# ###### [Link](https://python.langchain.com/docs/modules/data_connection/text_embedding/)

# %%
persist_directory = 'chroma_index'

# %%
# Current limitation of the 96 elements in input array
chunk_size = 96

# Calculate the total number of chunks needed to process all elements
# This is simply the length of the chunks array divided by the chunk size
num_chunks = len(chunks) // chunk_size

# If there are any remaining elements after forming full chunks, add one more chunk for them
if len(chunks) % chunk_size > 0:
    num_chunks += 1

for i in range(num_chunks):
    # Calculate the start index for the current chunk
    start_idx = i * chunk_size

    # Calculate the end index for the current chunk
    # This is the start index plus the chunk size, but it should not exceed the length of the chunks array
    end_idx = min(start_idx + chunk_size, len(chunks))

    # Slice the chunks array to get the current chunk
    current_chunk = chunks[start_idx:end_idx]

    # Process the current chunk
    vectordb = Chroma.from_documents(documents=current_chunk, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    vectordb = None


# #### Load the ChromaDB from a local file

# %%
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)


# ##### Get the DB to check for a valid file

# %%
vectordb.get()


# ## Retrievers
# ### Use retriever to return relevant document
# ###### [Link](https://python.langchain.com/docs/modules/data_connection/retrievers/)

# %%
retriever = vectordb.as_retriever()
query = "Who was the first CEO of Oracle?"
docs = vectordb.similarity_search(query)
print(docs)


# ### Run test querries to return documents

# %%
docs = retriever.get_relevant_documents("Who was the first CEO of Oracle?")
len(docs)


# ### Pass search [arguments](https://python.langchain.com/docs/modules/data_connection/retrievers/vectorstore)

# %%
retriever = vectordb.as_retriever(search_kwargs={"k": 2})
retriever.search_type


# ### Create a chain
# ###### [Link](https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html)


# #### [Optional] Create a handler to get verbose information. Helpful while troubleshooting

# %%
from langchain.globals import set_verbose, set_debug
set_debug(True)
set_verbose(True)
from langchain.callbacks import StdOutCallbackHandler
handler = StdOutCallbackHandler()


# #### RetrievalQA Chain with map_reduce as chain type. Enable callback variable to get verbose output

# %%
# create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type= "map_reduce",
                                  retriever=retriever,
                                  return_source_documents=True,
                                    #callbacks=[handler],
                                    )


# #### Function to return relevant output

# %%
def print_output(response):
    # Check if 'result' key exists in the response and print its value
    if 'result' in response:
        print(f"Result: {response['result']} \n\n")
    else:
        print("Result: No result found.\n\n")

    # Check if 'source_documents' key exists and it is a list
    if 'source_documents' in response and isinstance(response['source_documents'], list):
        # Iterate through each source document in the list
        for i, src in enumerate(response['source_documents'], start=1):
            # Access 'metadata' directly assuming 'src' is an object with a 'metadata' attribute
            # Check if 'metadata' exists and is a dictionary, then access 'source'
            if hasattr(src, 'metadata') and isinstance(src.metadata, dict):
                source_url = src.metadata.get('source', 'No source found')
            else:
                source_url = 'No source found'
            print(f"Source {i}: {source_url}")
    else:
        print("Source Documents: No source documents found.")

    return None


# ### Query

# %%
query = "When did Oracle partner with microsoft? Answer in one line"
llm_response = qa_chain.invoke(query)
print_output(llm_response)

# %%
query = "Who is the current CEO of Oracle?"
llm_response = qa_chain.invoke(query)
print_output(llm_response)

# %%
query = "What was the original name of Oracle?"
llm_response = qa_chain.invoke(query)
print_output(llm_response)

# %%
query = "What products does Oracle sell? Anser the question in bulltet points."
llm_response = qa_chain.invoke(query)
print_output(llm_response)

# %%
query = "When did Oracle come into an existence? Answer in one line"
llm_response = qa_chain.invoke(query)
print_output(llm_response)

# %%
query = "When did Oracle aquire Cerner?Answer in one line"
llm_response = qa_chain.invoke(query)
print_output(llm_response)

# %%
query = "Name few hardware products. The answer should be in bullet points"
llm_response = qa_chain.invoke(query)
print_output(llm_response)

# %%
query = '''
Could you analyze and discuss the ethical framework and values that guide Oracle Corporation? Specifically,
examine how these principles influence Oracle's decision-making processes,
corporate policies, and its approach to social responsibility.
Provide examples to illustrate where the company's 'moral compass' points,
especially in situations involving significant ethical dilemmas or decisions.
'''
llm_response = qa_chain.invoke(query)
print_output(llm_response)

# %%
query = '''
Please calculate the total amount Oracle has spent on acquisitions where the purchase price is publicly disclosed.
Exclude any acquisitions where the purchase price has not been shared.
Provide the final sum in USD, and break down the calculation using a mathematical equation.
Ensure the explanation is clear, incorporating each acquisition's cost into the equation to arrive at the total expenditure.
'''
llm_response = qa_chain.invoke(query)
print_output(llm_response)
