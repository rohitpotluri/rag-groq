# The data:
# Occupation, Earnings, and Job Characteristics: July 2022
# Household Income in States andMetropolitan Areas: 2022
# Poverty in States and Metropolitan Areas: 2022
# Health Insurance Coverage Status and Type by Geography: 2021 and 2022

# Download documents from U.S. Census Bureau to local directory.

from urllib.request import urlretrieve
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import GOOGLE_API
import os

os.makedirs("us_census", exist_ok=True)
files = [
    "https://www.census.gov/content/dam/Census/library/publications/2022/demo/p70-178.pdf",
    "https://www.census.gov/content/dam/Census/library/publications/2023/acs/acsbr-017.pdf",
    "https://www.census.gov/content/dam/Census/library/publications/2023/acs/acsbr-016.pdf",
    "https://www.census.gov/content/dam/Census/library/publications/2023/acs/acsbr-015.pdf",
]
for url in files:
    file_path = os.path.join("us_census", url.rpartition("/")[2])
    urlretrieve(url, file_path)


#Split documents to smaller chunks
# Load pdf files in the local directory
loader = PyPDFDirectoryLoader("./us_census/")

docs_before_split = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap  = 50,
)

docs_after_split = text_splitter.split_documents(docs_before_split)

embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API, model="models/embedding-001")

vectorstore = FAISS.from_documents(docs_after_split, embeddings)