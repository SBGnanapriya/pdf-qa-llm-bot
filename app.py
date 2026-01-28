import streamlit as st
import tempfile
import os
import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="PDF QA Bot (LLM)", layout="centered")
st.title("📄 PDF Question Answering Bot (LLM + RAG)")
st.write("Upload a PDF and ask questions from it.")


# -------------------------------
# Load LLM (NO pipeline)
# -------------------------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

tokenizer, model = load_llm()


# -------------------------------
# Load Embeddings
# -------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()


# -------------------------------
# Upload PDF
# -------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    st.success(f"PDF loaded: {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    question = st.text_input("Ask a question from the PDF")

    if question:
        results = vectorstore.similarity_search_with_score(question, k=3)

        relevant_docs = [
            doc for doc, score in results if score < 1.0
        ]

        if not relevant_docs:
            st.error("❌ Answer not found in the PDF.")
        else:
            context = "\n".join(doc.page_content for doc in relevant_docs)

            prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say "Answer not found".

Context:
{context}

Question:
{question}
"""

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            outputs = model.generate(
                **inputs,
                max_new_tokens=200
            )

            answer = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            st.subheader("✅ Answer")
            st.write(answer)

    os.remove(pdf_path)
