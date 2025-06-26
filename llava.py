import os
import tempfile
import shutil
import logging
import streamlit as st
import pandas as pd
import base64
import requests
import json
import uuid
from datetime import datetime
from PIL import Image
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DB_DIR = "./faiss_db"
CHAT_DIR = "./chat_sessions"

logging.basicConfig(level=logging.INFO, filename="app.log", format="%(asctime)s [%(levelname)s] %(message)s")

st.set_page_config(page_title="PDF QA with Tables", layout="wide")
st.title("📄 PDF Text & Table Extractor + Chat QA")

# --- LLaVA-based Extraction ---
def extract_with_llava(pdf_path):
    try:
        images = convert_from_path(pdf_path)
        results = []
        total = len(images)
        progress = st.progress(0)

        def process_page(index, img):
            img = img.resize((768, int(img.height * 768 / img.width)))
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
                img.save(temp_img.name)
                with open(temp_img.name, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")
            prompt = "Extract all tables and text from this document image. Return tables in CSV format followed by the text."
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": "llava:latest", "prompt": prompt, "images": [img_base64], "stream": False},
                timeout=180
            )
            os.unlink(temp_img.name)
            result = response.json()
            return result.get("response", "")

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_page, i, img): i for i, img in enumerate(images)}
            for i, future in enumerate(as_completed(futures)):
                try:
                    results.append(future.result())
                except Exception as e:
                    logging.error(f"LLaVA error on page {futures[future]}: {e}")
                progress.progress((i + 1) / total)

        return "\n\n".join(results)
    except Exception as e:
        logging.error(f"LLaVA extraction failed: {e}")
        st.error(f"LLaVA extraction failed: {e}")
        return ""

@st.cache_resource(show_spinner=False)
def load_and_index(files):
    all_docs = []
    with tempfile.TemporaryDirectory() as td:
        for file in files:
            path = os.path.join(td, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            try:
                loader = PyPDFLoader(path)
                all_docs.extend(loader.load())
                llava_text = extract_with_llava(path)
                all_docs.append(Document(page_content=llava_text, metadata={"source": file.name}))
            except Exception as e:
                logging.error(f"Failed to process {file.name}: {e}")
                st.error(f"Failed to process {file.name}: {e}")

    if not all_docs:
        st.warning("No documents were successfully loaded or extracted.")
        return None

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(all_docs)
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(DB_DIR)
        st.success("✅ Documents processed and indexed successfully!")
        return vs
    except Exception as e:
        logging.error(f"FAISS indexing error: {e}")
        st.error(f"FAISS indexing error: {e}")
        return None

def load_existing_index():
    if not os.path.exists(DB_DIR):
        return None
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        return FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logging.error(f"Failed to load existing FAISS DB: {e}")
        st.error(f"Failed to load existing FAISS DB: {e}")
        return None

def get_chat_chain(vs):
    prompt = ChatPromptTemplate.from_template("You are a table analysis expert.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    return {"context": vs.as_retriever(), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

def clear_db():
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        logging.info(f"FAISS DB directory '{DB_DIR}' cleared.")

# --- UI Sidebar ---
with st.sidebar:
    st.image("img/ACL_Digital.png", width=180)
    st.image("img/Cipla_Foundation.png", width=180)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.header("📂 Upload PDFs")
    uploaded = st.file_uploader("Select PDFs", type="pdf", accept_multiple_files=True)
    run = st.button("📊 Extract & Index")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.header("🛠 Control")
    if st.button("🗑 Clear DB"):
        clear_db()
        st.session_state.vs = None
        st.success("DB cleared")
    if st.button("🧹 Clear Chat"):
        st.session_state.msgs = []
        st.success("Chat cleared")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.header("💬 Chat History")
    os.makedirs(CHAT_DIR, exist_ok=True)

    def summarize_chat(msgs):
        for msg in msgs:
            if msg["role"] == "user" and msg["content"].strip():
                return msg["content"].strip().split("\n")[0][:40].replace(" ", "_").replace("?", "").replace(":", "").lower()
        return f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if "chat_id" not in st.session_state:
        base = summarize_chat(st.session_state.get("msgs", []))
        st.session_state.chat_id = f"{base}_{uuid.uuid4().hex[:4]}"
        with open(os.path.join(CHAT_DIR, f"{st.session_state.chat_id}.json"), "w") as f:
            json.dump([], f)

    if st.button("➕ New Chat"):
        base = summarize_chat(st.session_state.get("msgs", []))
        st.session_state.chat_id = f"{base}_{uuid.uuid4().hex[:4]}"
        st.session_state.msgs = []
        with open(os.path.join(CHAT_DIR, f"{st.session_state.chat_id}.json"), "w") as f:
            json.dump([], f)
        st.rerun()

    session_files = sorted([f for f in os.listdir(CHAT_DIR) if f.endswith(".json")], key=lambda x: os.path.getmtime(os.path.join(CHAT_DIR, x)), reverse=True)[:10]

    for fname in session_files:
        label = fname.replace(".json", "").replace("_", " ").title()
        if st.button(f"💬 {label}"):
            st.session_state.chat_id = fname.replace(".json", "")
            with open(os.path.join(CHAT_DIR, fname), "r") as f:
                st.session_state.msgs = json.load(f)
            st.session_state.vs = load_existing_index()
            st.rerun()

# --- Main ---
if "vs" not in st.session_state:
    st.session_state.vs = load_existing_index()
if "msgs" not in st.session_state:
    st.session_state.msgs = []

if run and uploaded:
    st.session_state.msgs = []
    with st.spinner("Processing documents and building index..."):
        st.session_state.vs = load_and_index(uploaded)
    if st.session_state.vs:
        st.session_state.msgs.append({"role": "assistant", "content": "Extraction & indexing done. Ask anything!"})

for msg in st.session_state.msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask about the PDF content or tables..."):
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
        st.error("Please upload and process PDFs first to enable chat functionality.")

# --- Save Chat ---
if "chat_id" in st.session_state:
    with open(os.path.join(CHAT_DIR, f"{st.session_state.chat_id}.json"), "w") as f:
        json.dump(st.session_state.msgs, f)