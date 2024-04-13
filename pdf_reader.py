from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# Importing necessary libraries from Transformers
from transformers import pipeline

st.set_page_config(page_title="DocGenius: Document Generation AI")
st.header("Ask Your PDF📄")
pdf = st.file_uploader("Upload your PDF", type="pdf")
if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )  

    chunks = text_splitter.split_text(text)

    # Remove the OpenAIEmbeddings and FAISS parts, since we're not using OpenAI
    # embeddings and FAISS in this version.

    query = st.text_input("Ask your Question about your PDF")
    if query:
        # Instead of OpenAI model, we'll use a pre-trained model from Hugging Face
        # Load a pre-trained model and tokenizer
        qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased")

        # Iterate through chunks to find the answer to the query
        found_answer = False
        for chunk in chunks:
            result = qa_pipeline(question=query, context=chunk)
            if result["score"] > 0.5:  # You can adjust the threshold as needed
                st.success(f"Answer: {result['answer']}")
                found_answer = True
                break

        if not found_answer:
            st.warning("Sorry, I couldn't find an answer to your question in the document.")
