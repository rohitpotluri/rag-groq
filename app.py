"""
Main application for Document Q&A with GROQ and Google Embeddings
"""
import time
import PyPDF2
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from config import GROQ_API
from prompt import get_prompt_template
from embeddings import generate_embeddings

st.title("Document Q&A with GROQ and Google Embeddings")

llm = ChatGroq(groq_api_key=GROQ_API, model_name="gemma-7b-it")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
# Initialize vectorstore in session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if st.button("Generate Embeddings"):
    if uploaded_file is not None:
        reader = PyPDF2.PdfReader(uploaded_file)
        PDF_TEXT = ""
        for page in reader.pages:
            PDF_TEXT += page.extract_text()
            st.session_state.vectorstore = generate_embeddings(PDF_TEXT)
        st.write("Embeddings are ready and stored in the vector database!")
    else:
        st.error("Please upload a PDF file to generate embeddings.")

with st.form("my_form"):
    user_query = st.text_input("Enter your question based on the documentation:")
    submitted = st.form_submit_button("Send")

if submitted:
    if st.session_state.vectorstore:
        document_chain = create_stuff_documents_chain(llm, get_prompt_template())
        retriever = st.session_state.vectorstore.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with st.spinner("Retrieving and processing your query..."):
            start_time = time.process_time()
            response = retrieval_chain.invoke({'input': user_query})
            end_time = time.process_time() - start_time

            st.subheader("Answer:")
            st.write(response['answer'])

            st.write(f"Response time: {end_time:.2f} seconds")

            with st.expander("Relevant Document Chunks"):
                for i, doc in enumerate(response["context"]):
                    st.write(f"Chunk {i + 1}: {doc.page_content}")
                    st.write("---")
    else:
        st.error('Please upload a file and click on generate' +
                 ' embeddings first to answer your queries.')
