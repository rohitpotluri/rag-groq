"""
Generates embeddings by calling GoogleGenerativeAIEmbeddings
"""

# Third-party library imports from langchain and langchain_community
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Project-specific imports (if applicable)
from config import GOOGLE_API

def generate_embeddings(text):
    """
    Returns a vectorstore based on the content in attached PDF files.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 700,
        chunk_overlap  = 50,
    )
    document = Document(page_content=text)
    docs_after_split = text_splitter.split_documents([document])
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API,
                                              model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs_after_split, embeddings)

    return vectorstore
