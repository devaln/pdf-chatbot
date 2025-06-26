# app.py — PDF QA with LLaVA extraction + LLaMA3 chat

import os
import tempfile
import shutil
import logging
import streamlit as st
import fitz  # PyMuPDF
import requests
import json
import uuid
from datetime import datetime
from PIL import Image
import base64
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

# --- Config ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3:latest"
OLLAMA_VISION_MODEL = "llava:latest"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DB_DIR = "./faiss_db"
CHAT_DIR = "./chat_sessions"

logging.basicConfig(level=logging.INFO, filename="app.log", format="%(asctime)s [%(levelname)s] %(message)s")

st.set_page_config(page_title="PDF QA with Tables", layout="wide")
st.title("📄 PDF Text & Table Extractor + Chat QA")

# --- Extract Text & Tables using LLaVA ---
def extract_with_llava(images):
    extracted_chunks = []
    for i, img in enumerate(images):
        buf = BytesIO()
        img = img.convert("RGB")
        img.thumbnail((768, 768))
        img.save(buf, format="PNG")
        b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = "You are a document understanding AI. Extract all text and tables from this image. Present tables as CSV."

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": OLLAMA_VISION_MODEL, "prompt": prompt, "images": [b64_img], "stream": False},
                timeout=180
            )
            result = response.json()
            content = result.get("response", "")
            extracted_chunks.append(content)
            logging.info(f"LLaVA page {i+1} extracted")
        except Exception as e:
            logging.error(f"LLaVA extraction failed on page {i+1}: {e}")
            extracted_chunks.append("")

    return "\n\n".join(extracted_chunks)

# --- Indexing Logic ---
@st.cache_resource(show_spinner=False)
def load_and_index(files):
    all_docs = []
    with tempfile.TemporaryDirectory() as td:
        for file in files:
            path = os.path.join(td, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())

            try:
                images = convert_from_path(path)
                with st.spinner(f"Extracting {file.name} with LLaVA..."):
                    progress = st.progress(0)
                    with ThreadPoolExecutor() as executor:
                        chunks = list(executor.map(lambda i: extract_with_llava([images[i]]), range(len(images))))
                        text = "\n".join(chunks)
                        progress.progress(100)

                all_docs.append(Document(page_content=text, metadata={"source": file.name}))
            except Exception as e:
                logging.error(f"Failed on {file.name}: {e}")
                st.error(f"Failed to extract {file.name}: {e}")

    if not all_docs:
        st.warning("No documents were loaded.")
        return None

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(all_docs)
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(DB_DIR)
        st.success("✅ Documents indexed successfully!")
        return vs
    except Exception as e:
        st.error(f"FAISS indexing error: {e}")
        return None

# --- Chat Chain ---
def get_chat_chain(vs):
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use this context to answer:

{context}

Question: {question}

Answer:
""")
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)
    return {"context": vs.as_retriever(), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# --- DB Clear ---
def clear_db():
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        st.success("DB cleared")

# --- Sidebar UI ---
with st.sidebar:
    st.header("📂 Upload PDFs")
    uploaded = st.file_uploader("Select PDFs", type="pdf", accept_multiple_files=True)
    run = st.button("📊 Extract & Index")
    if st.button("🗑 Clear DB"):
        clear_db()
        st.session_state.vs = None
    if st.button("🔁 Clear Chat"):
        st.session_state.msgs = []

    st.markdown("---")
    st.header("💬 Chat History")
    os.makedirs(CHAT_DIR, exist_ok=True)

    def summarize_chat(msgs):
        for msg in msgs:
            if msg["role"] == "user":
                return msg["content"].strip().split()[0:5]
        return f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if "chat_id" not in st.session_state:
        base = "_".join(summarize_chat(st.session_state.get("msgs", [])))
        st.session_state.chat_id = f"{base}_{uuid.uuid4().hex[:4]}"
        with open(os.path.join(CHAT_DIR, f"{st.session_state.chat_id}.json"), "w") as f:
            json.dump([], f)

    if st.button("➕ New Chat"):
        base = "_".join(summarize_chat(st.session_state.get("msgs", [])))
        st.session_state.chat_id = f"{base}_{uuid.uuid4().hex[:4]}"
        st.session_state.msgs = []
        with open(os.path.join(CHAT_DIR, f"{st.session_state.chat_id}.json"), "w") as f:
            json.dump([], f)
        st.rerun()

    session_files = sorted(
        [f for f in os.listdir(CHAT_DIR) if f.endswith(".json")],
        key=lambda x: os.path.getmtime(os.path.join(CHAT_DIR, x)),
        reverse=True
    )[:10]

    for fname in session_files:
        label = fname.replace(".json", "").replace("_", " ").title()
        if st.button(f"💬 {label}"):
            st.session_state.chat_id = fname.replace(".json", "")
            with open(os.path.join(CHAT_DIR, fname), "r") as f:
                st.session_state.msgs = json.load(f)
            st.session_state.vs = FAISS.load_local(DB_DIR, OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL), allow_dangerous_deserialization=True)
            st.rerun()

# --- Main UI Logic ---
if "vs" not in st.session_state:
    if os.path.exists(DB_DIR):
        st.session_state.vs = FAISS.load_local(DB_DIR, OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL), allow_dangerous_deserialization=True)
if "msgs" not in st.session_state:
    st.session_state.msgs = []

if run and uploaded:
    st.session_state.msgs = []
    with st.spinner("Processing documents and building index..."):
        st.session_state.vs = load_and_index(uploaded)
    if st.session_state.vs:
        st.session_state.msgs.append({"role": "assistant", "content": "Extraction & indexing complete. Ask your questions!"})

for msg in st.session_state.msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask something about the PDFs..."):
    st.session_state.msgs.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.vs:
        chain = get_chat_chain(st.session_state.vs)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = "".join(chain.stream(query))
                st.markdown(resp)
                st.session_state.msgs.append({"role": "assistant", "content": resp})
    else:
        st.error("Upload and process PDFs first.")

# --- Save chat ---
if "chat_id" in st.session_state:
    with open(os.path.join(CHAT_DIR, f"{st.session_state.chat_id}.json"), "w") as f:
        json.dump(st.session_state.msgs, f)