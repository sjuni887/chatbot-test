import streamlit as st
from transformers import pipeline
import fitz  # PyMuPDF
import pandas as pd

# Load the GPT-2 model
@st.cache_resource
def load_model():
    return pipeline('text-generation', model='gpt2')

generator = load_model()

# Function to read text from PDF
def read_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Function to read text from CSV
def read_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

# Function to generate a response from the model
def generate_response(prompt):
    response = generator(prompt, max_length=150)
    return response[0]['generated_text']

# Streamlit UI
st.title('PDF and CSV Reading Chatbot')

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".pdf"):
        file_text = read_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".csv"):
        file_text = read_csv(uploaded_file)
    
    st.write("File content:")
    st.text(file_text)

    user_input = st.text_input("Ask a question about the content:")
    if user_input:
        response = generate_response(user_input + "\n" + file_text)
        st.write("Chatbot response:")
        st.text(response)
