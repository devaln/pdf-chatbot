# --- Imports ---
import os
import tempfile
import shutil
import logging
import streamlit as st
import pandas as pd
from PIL import Image
import fitz  # PyMuPDF
import pdfplumber
import camelot
import pytesseract
from pdf2image import convert_from_path

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
OLLAMA_LLM_MODEL = "llama4:latest"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DB_DIR = "./faiss_db"

logging.basicConfig(level=logging.INFO, filename="app.log", format="%(asctime)s [%(levelname)s] %(message)s")

st.set_page_config(page_title="PDF QA with Tables", layout="wide")
st.title("📄 PDF Text & Table Extractor + Chat QA")

# --- Utility Functions ---
def clean_df(df):
    df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
    return df.fillna("")

def extract_scanned_pdf_with_ocr(pdf_path, llm):
    try:
        images = convert_from_path(pdf_path)
        ocr_full_text = ""
        for i, img in enumerate(images):
            page_text = pytesseract.image_to_string(img)
            ocr_full_text += f"--- Page {i+1} ---\n{page_text}\n\n"

        ocr_full_text = ocr_full_text.strip()
        if not ocr_full_text:
            return "", ""

        prompt_clean = f"""You are a document cleaner.
From the following OCR text, remove any tables or structured data.
Only return the clean paragraph-like body text for QA and summarization.

OCR Text:
{ocr_full_text}
"""
        clean_text = llm.invoke(prompt_clean).content if prompt_clean.strip() else ""

        prompt_table = f"""You are a table understanding expert.
Extract all tables from the following OCR text and convert them to CSV format.
Ensure each table is clearly separated and labeled.

OCR Text:
{ocr_full_text}
"""
        table_csv = llm.invoke(prompt_table).content if prompt_table.strip() else ""

        final = clean_text.strip()
        if table_csv and "No tables found" not in table_csv:
            final += "\n\nOCR LLM-Extracted Tables:\n" + table_csv.strip()

        return final.strip(), clean_text.strip()

    except Exception as e:
        logging.error(f"OCR + LLM extraction failed: {e}")
        st.error(f"OCR + LLM failed: {e}")
        return "", ""

def extract_all_tables(pdf_path, scanned_mode=False, llm=None):
    if scanned_mode:
        return extract_scanned_pdf_with_ocr(pdf_path, llm)

    dfs = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        dfs.append(clean_df(df))
    except Exception as e:
        logging.warning(f"pdfplumber failed: {e}")

    try:
        camelot_tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
        for t in camelot_tables:
            df = t.df
            if df.shape[0] > 1 and df.shape[1] > 1:
                dfs.append(clean_df(df))
    except Exception as e:
        logging.warning(f"camelot failed: {e}")

    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
    except Exception as e:
        logging.error(f"PDF text extraction failed: {e}")
        text = ""

    prompt = f"You are a table understanding expert.\n\nExtract all tables and convert to CSV:\n\n{text}"
    llm_csv = llm.invoke(prompt).content if prompt.strip() else ""

    table_texts = [f"Table {i+1}:\n{df.to_csv(index=False)}" for i, df in enumerate(dfs)]
    table_texts.append("LLM-Structured Tables:\n" + llm_csv)
    return "\n\n".join(table_texts), text

@st.cache_resource(show_spinner=False)
def load_and_index(files, scanned_mode=False):
    all_docs = []
    with tempfile.TemporaryDirectory() as td:
        llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=300)
        for file in files:
            path = os.path.join(td, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())

            if os.path.getsize(path) == 0:
                st.warning(f"⚠ Skipping {file.name} — file is empty or not fully uploaded.")
                continue

            try:
                loader = PyPDFLoader(path)
                all_docs.extend(loader.load())

                text_csv, raw_text = extract_all_tables(path, scanned_mode, llm)
                if not (text_csv or raw_text):
                    st.warning(f"⚠ Skipping {file.name} — no valid content extracted.")
                    continue

                all_docs.append(Document(page_content=text_csv + "\n" + raw_text, metadata={"source": file.name}))
            except Exception as e:
                logging.error(f"Failed to process {file.name}: {e}")
                st.error(f"Failed to process {file.name}: {e}")

    if not all_docs:
        st.warning("⚠ No documents were successfully loaded or extracted.")
        return None

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(all_docs)
    if not chunks:
        st.error("❌ No content extracted to index. Check your PDFs or OCR.")
        return None

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

# --- Sidebar ---
with st.sidebar:
    st.header("📂 Upload PDFs")
    uploaded = st.file_uploader("Select PDFs", type="pdf", accept_multiple_files=True)
    scanned_mode = st.checkbox("📸 PDF is scanned (image only)?")
    run = st.button("📊 Extract & Index")
    st.markdown(""" <hr> """, unsafe_allow_html=True)
    st.header("🛠 Control")
    if st.button("🗑 Clear DB"):
        clear_db()
        st.session_state.vs = None
        st.success("DB cleared")
    if st.button("🧹 Clear Chat"):
        st.session_state.msgs = []
        st.success("Chat cleared")

# --- Main Interface ---
if "vs" not in st.session_state:
    st.session_state.vs = load_existing_index()
if "msgs" not in st.session_state:
    st.session_state.msgs = []

if run and uploaded:
    st.session_state.msgs = []
    with st.spinner("Processing documents and building index..."):
        st.session_state.vs = load_and_index(uploaded, scanned_mode)
    if st.session_state.vs:
        st.session_state.msgs.append({"role": "assistant", "content": "✅ Extraction & indexing done. Ask anything!"})

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