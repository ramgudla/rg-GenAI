
import oracledb
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from PyPDF2 import PdfReader

from .oci_config import (auth_profile, compartment_id,
                              embeddings_model_id, embeddings_model_kwargs,
                              llm_model_id, llm_model_kwargs, service_endpoint)

DOCUMENTS_PATH = "data/budget_speech.pdf"

class GenAIWrapper:
    def __init__(self):
        self.compartment_id = compartment_id
        self.auth_profile = auth_profile
        self.service_endpoint = service_endpoint
        self.embeddings_model_id = embeddings_model_id
        self.llm_model_id = llm_model_id
        self.llm_model_kwargs = llm_model_kwargs
        self.embeddings_model_kwargs = embeddings_model_kwargs
        self.ora23ai_table_name = "ora23ai_index"
        self.chroma_persist_directory = "chroma_index"
        self.faiss_index = "faiss_index"
        # Initialize both models
        self.initialize_embeddings()
        self.initialize_llm()

    def initialize_embeddings(self):
        self.embeddings = OCIGenAIEmbeddings(
            model_id=self.embeddings_model_id,
            service_endpoint=self.service_endpoint,
            compartment_id=self.compartment_id,
            auth_type="API_KEY",
            auth_profile=self.auth_profile,
            model_kwargs=self.embeddings_model_kwargs
        )

    def initialize_llm(self):
        self.llm = ChatOCIGenAI(
            model_id=self.llm_model_id,
            service_endpoint=self.service_endpoint,
            compartment_id=self.compartment_id,
            auth_type="API_KEY",
            auth_profile=self.auth_profile,
            model_kwargs=self.llm_model_kwargs
        )

    def persist_ora23ai_vs(self):
        self.initialize_db_connection()
        # creating a pdf reader object
        pdf = PdfReader(DOCUMENTS_PATH)

        # print number of pages in pdf file
        print("The number of pages in this document is ", len(pdf.pages))

        # print the first page
        print(pdf.pages[0].extract_text())

        if pdf is not None:
            print("Transforming the PDF document to text...")
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        print("You have transformed the PDF document to text format")

        # Chunk the text document into smaller chunks
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=800, chunk_overlap=100, length_function=len)
        chunks = text_splitter.split_text(text)
        print(chunks[0])

        """
        Converts a row from a DataFrame into a Document object suitable for ingestion into Oracle Vector Store.
        - row (dict): A dictionary representing a row of data with keys for 'id', 'link', and 'text'.
        """
        docs_for_oracle_vs = [OCIModelWrapper.chunks_to_docs_wrapper({'id': page_num, 'link': f'Page {page_num}', 'text': text})
                              for page_num, text in enumerate(chunks)]

        username = "testuser"
        password = "testuser"
        dsn = "ramgudla.ad2.devintegratiphx.oraclevcn.com:1521/FREEPDB1"

        # Connect to the database
        try:
            conn23ai = oracledb.connect(
                user=username, password=password, dsn=dsn)
            print("ora23ai Connection successful!")
        except Exception as e:
            print("ora23ai Connection failed!")

        # Configure the vector store with the model, table name, and using the indicated distance strategy for the similarity search and vectorize the chunks
        oracle_vs = OracleVS.from_documents(
            docs_for_oracle_vs, self.embeddings,
            client=conn23ai,
            table_name=self.ora23ai_table_name,
            distance_strategy=DistanceStrategy.DOT_PRODUCT
        )
        oracle_vs = None

    def persist_chroma_vs(self):
        # Load documents
        loader = PyPDFLoader(DOCUMENTS_PATH)
        docs_for_chroma_vs = loader.load_and_split()

        chroma_vs = Chroma.from_documents(
            docs_for_chroma_vs,
            embedding=self.embeddings,
            persist_directory=self.chroma_persist_directory
        )
        chroma_vs.persist()
        chroma_vs = None

    def persist_faiss_vs(self):
        # Load documents
         # creating a pdf reader object
        pdf = PdfReader(DOCUMENTS_PATH)

        # print number of pages in pdf file
        print("The number of pages in this document is ", len(pdf.pages))

        # print the first page
        print(pdf.pages[0].extract_text())

        if pdf is not None:
            print("Transforming the PDF document to text...")
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        print("You have transformed the PDF document to text format")

        # Chunk the text document into smaller chunks
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=800, chunk_overlap=100, length_function=len)
        chunks = text_splitter.split_text(text)
        print(chunks[0])

        faiss_vs = FAISS.from_texts(chunks, self.embeddings)
        faiss_vs.save_local(self.faiss_index)
        faiss_vs = None

    def get_ora23ai_vs(self) -> OracleVS:
        self.initialize_db_connection()
        oracle_vs = OracleVS(client=self.ora23aiConn,
                             embedding_function=self.embeddings,
                             table_name=self.ora23ai_table_name,
                             distance_strategy=DistanceStrategy.DOT_PRODUCT,
        )
        return oracle_vs

    def get_chroma_vs(self) -> Chroma:
        chroma_vs = Chroma(persist_directory=self.chroma_persist_directory,
                           embedding_function=self.embeddings)
        return chroma_vs
    
    def get_faiss_vs(self) -> FAISS:
        faiss_vs = FAISS.load_local(self.faiss_index, self.embeddings, allow_dangerous_deserialization=True)
        return faiss_vs
    
    def initialize_db_connection(self):
        username = "testuser"
        password = "testuser"
        dsn = "ramgudla.ad2.devintegratiphx.oraclevcn.com:1521/FREEPDB1"

        # get db connection
        try:
            conn23ai = oracledb.connect(
                user=username, password=password, dsn=dsn)
            print("Connection successful!\n")
        except Exception as e:
            print("Connection failed!\n")
        self.ora23aiConn = conn23ai

    # Function to format and add metadata to Oracle 23ai Vector Store
    @staticmethod
    def chunks_to_docs_wrapper(row: dict) -> Document:
        """
        Converts text into a Document object suitable for ingestion into Oracle Vector Store.
        - row (dict): A dictionary representing a row of data with keys for 'id', 'link', and 'text'.
        """
        metadata = {'id': str(row['id']), 'link': row['link']}
        return Document(page_content=row['text'], metadata=metadata)

    @staticmethod
    def pretty_print_docs(docs):
        print(
            f"\n{'-' * 100}\n".join(
                [f"Document {i+1}:\n\n" +
                    d.page_content for i, d in enumerate(docs)]
            )
        )