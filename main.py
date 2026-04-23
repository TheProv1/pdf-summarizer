import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq

def summarize_pdf(file_path, user_api):
    print(f"Loading file {file_path}")

    loader = PyPDFLoader(file_path=file_path)
    pages = loader.load()

    text_split = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    
    docs = text_split.split_documents(pages)

    llm = ChatGroq(model = 'llama-3.1-8b-instant',
                   temperature=0,
                   api_key=user_api)

    chain = load_summarize_chain(llm)
    result = chain.invoke(input = docs, chain_type = "map_reduce")

    return result["output_text"]


st.set_page_config(page_title="Document Summarizer")

with st.sidebar:
    st.header("Config")
    api_key_input = st.text_input("Enter your Groq API Key: ", type="password")
    
st.title("Document Summarizer")
st.write("Upload a PDF file and get an instant summary")

uploaded_file = st.file_uploader("Upload file", type = ["pdf"])
if uploaded_file is not None:
    if st.button("Generate Summary"):
        if api_key_input is None:
            st.warning("Please enter your Groq API Key to continue")
            st.stop()

        with st.spinner("Reading file and summarizing"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_filepath = temp_file.name

            try:
                summary = summarize_pdf(temp_filepath, api_key_input)
                st.write(summary)

            except Exception as e:
                st.error(f"Error {e}")
            
            finally:
                os.remove(temp_filepath)
