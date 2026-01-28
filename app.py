import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import tempfile
import os

# ---------------- UI ----------------
st.set_page_config(page_title="PDF Q&A Bot", layout="centered")
st.title("📄 PDF Question Answering Bot (LLM)")
st.write("Upload a PDF and ask questions from it.")

# ---------------- Upload PDF ----------------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # Embeddings + Vector DB
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    st.success("✅ PDF processed successfully!")

    # ---------------- Question ----------------
    question = st.text_input("Ask a question from the PDF")

    if question:
        docs_scores = vectorstore.similarity_search_with_score(question, k=3)

        THRESHOLD = 0.7
        relevant_docs = [
            doc for doc, score in docs_scores if score < THRESHOLD
        ]

        if not relevant_docs:
            st.error("❌ Answer not found in the PDF.")
        else:
            context = "\n".join([doc.page_content for doc in relevant_docs])

            # Load LLM
            llm = pipeline(
                "text2text-generation",
                model="google/flan-t5-base"
            )

            prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say "Answer not found".

Context:
{context}

Question:
{question}
"""

            answer = llm(prompt, max_length=200)[0]["generated_text"]
            st.subheader("✅ Answer")
            st.write(answer)

    os.remove(pdf_path)
