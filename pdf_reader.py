from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

# Importing necessary libraries from Transformers
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

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

    query = st.text_input("Ask your Question about your PDF")
    if query:
        # Load pre-trained model and tokenizer
        model_name = "distilbert-base-cased-distilled-squad"
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Iterate through chunks to find the answer to the query
        found_answer = False
        for chunk in chunks:
            inputs = tokenizer(query, chunk, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits

            # Get the most likely answer
            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1
            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

            if answer and answer != '[CLS]':
                st.success(f"Answer: {answer}")
                found_answer = True
                break

        if not found_answer:
            st.warning("Sorry, I couldn't find an answer to your question in the document.")
