"""
Main application for Document Q&A with GROQ and Google Embeddings
"""
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from config import GROQ_API
from embeddings import vectorstore
from prompt import get_prompt_template

st.title("Document Q&A with GROQ and Google Embeddings")

llm = ChatGroq(groq_api_key=GROQ_API, model_name="gemma-7b-it")

user_query = st.text_input("Enter your question based on the documentation:")

if st.button("Generate Embeddings"):
    st.write("Embeddings are ready and stored in the vector database!")

if user_query:
    document_chain = create_stuff_documents_chain(llm, get_prompt_template())
    retriever = vectorstore.as_retriever()
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
