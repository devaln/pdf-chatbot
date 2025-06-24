# # app.py (OCR-only, no Donut)

# import os
# import tempfile
# import shutil
# import logging
# import streamlit as st
# import pandas as pd
# import numpy as np
# from PIL import Image
# import fitz  # PyMuPDF
# import pdfplumber
# import camelot
# import torch
# import requests
# import pytesseract
# from pdf2image import convert_from_path

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.chat_models import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.documents import Document
# from langchain_core.runnables import RunnablePassthrough

# # --- Config ---
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# OLLAMA_BASE_URL = "http://localhost:11434"
# OLLAMA_LLM_MODEL = "llama3:latest"
# OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
# DB_DIR = "./faiss_db"

# logging.basicConfig(level=logging.INFO, filename="app.log", format="%(asctime)s [%(levelname)s] %(message)s")

# st.set_page_config(page_title="PDF QA with Tables", layout="wide")
# st.markdown("""
#     <style>
#         section[data-testid="stSidebar"] {
#             background-color: white !important;
#             border-right: 2px solid #e0e0e0 !important;
#         }
#     </style>
# """, unsafe_allow_html=True)
# st.title("\U0001F4C4 PDF Text & Table Extractor + Chat QA")

# # --- Helpers ---
# def clean_df(df):
#     df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
#     return df.fillna("")

# def extract_tables_pdfplumber(pdf_path):
#     dfs = []
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             for page in pdf.pages:
#                 tbls = page.extract_tables()
#                 for table in tbls:
#                     if table:
#                         df = pd.DataFrame(table[1:], columns=table[0])
#                         dfs.append(clean_df(df))
#     except Exception as e:
#         logging.warning(f"pdfplumber failed for {os.path.basename(pdf_path)}: {e}")
#     return dfs

# def extract_tables_camelot(pdf_path):
#     dfs = []
#     for flavor in ["lattice", "stream"]:
#         try:
#             tables = camelot.read_pdf(pdf_path, pages='all', flavor=flavor)
#             for t in tables:
#                 df = t.df
#                 if df.shape[0] > 1 and df.shape[1] > 1:
#                     dfs.append(clean_df(df))
#         except Exception as e:
#             logging.warning(f"camelot {flavor} failed for {os.path.basename(pdf_path)}: {e}")
#     return dfs

# def extract_scanned_pdf_with_ocr(pdf_path):
#     try:
#         images = convert_from_path(pdf_path)
#         full_text = ""
#         for img in images:
#             text = pytesseract.image_to_string(img)
#             full_text += text + "\n"

#         llm_prompt = f"""You are a table understanding expert.
# Extract all tables from the following OCR text and convert them to CSV format:

# {full_text}

# Only return CSV-formatted tables."""

#         response = requests.post(
#             url=f"{OLLAMA_BASE_URL}/api/generate",
#             json={"model": OLLAMA_LLM_MODEL, "prompt": llm_prompt, "stream": False},
#             timeout=120
#         )
#         result = response.json()
#         csv_text = result.get("response", "")

#         # st.subheader("LLM‑Structured Tables from OCR")
#         # st.text(csv_text)
#         return csv_text, full_text
#     except Exception as e:
#         logging.error(f"OCR + LLM extraction failed: {e}")
#         st.error(f"OCR + LLM failed: {e}")
#         return "", ""

# def extract_all_tables(pdf_path, scanned_mode=False):
#     if scanned_mode:
#         return extract_scanned_pdf_with_ocr(pdf_path)

#     dfs = extract_tables_pdfplumber(pdf_path)
#     dfs += extract_tables_camelot(pdf_path)

#     try:
#         doc = fitz.open(pdf_path)
#         text = "\n".join([page.get_text() for page in doc])
#     except Exception as e:
#         logging.error(f"PDF text extraction failed: {e}")
#         text = ""

#     prompt = f"You are a table understanding expert.\n\nExtract all tables from the following document and convert them to CSV format:\n\n{text}\n\nOnly return CSV-formatted tables."

#     try:
#         response = requests.post(
#             url=f"{OLLAMA_BASE_URL}/api/generate",
#             json={"model": OLLAMA_LLM_MODEL, "prompt": prompt, "stream": False},
#             timeout=120
#         )
#         result = response.json()
#         llm_csv = result.get("response", "")
#     except Exception as e:
#         logging.error(f"LLM extraction failed: {e}")
#         llm_csv = ""

#     table_texts = []
#     for i, df in enumerate(dfs):
#         st.subheader(f"Table {i+1} (Raw)")
#         st.dataframe(df)
#         table_texts.append(f"Table {i+1}:\n{df.to_csv(index=False)}")

#     # st.subheader("LLM‑Structured Tables")
#     # st.text(llm_csv)
#     table_texts.append("LLM-Structured Tables:\n" + llm_csv)

#     return "\n\n".join(table_texts), text

# @st.cache_resource(show_spinner=False)
# def load_and_index(files, scanned_mode=False):
#     all_docs = []
#     with tempfile.TemporaryDirectory() as td:
#         for file in files:
#             path = os.path.join(td, file.name)
#             with open(path, "wb") as f:
#                 f.write(file.getbuffer())
#             try:
#                 # st.info(f"📄 Loading {file.name}...")
#                 loader = PyPDFLoader(path)
#                 all_docs.extend(loader.load())
#                 # st.info("🔍 Extracting tables...")
#                 text_csv, raw_text = extract_all_tables(path, scanned_mode)
#                 all_docs.append(Document(page_content=text_csv + "\n" + raw_text, metadata={"source": file.name}))
#             except Exception as e:
#                 logging.error(f"Failed to process {file.name}: {e}")
#                 st.error(f"Failed to process {file.name}: {e}")

#     if not all_docs:
#         st.warning("No documents were successfully loaded or extracted.")
#         return None

#     chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(all_docs)
#     try:
#         embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
#         vs = FAISS.from_documents(chunks, embeddings)
#         vs.save_local(DB_DIR)
#         st.success("✅ Documents processed and indexed successfully!")
#         return vs
#     except Exception as e:
#         logging.error(f"FAISS indexing error: {e}")
#         st.error(f"FAISS indexing error: {e}")
#         return None

# def load_existing_index():
#     if not os.path.exists(DB_DIR):
#         return None
#     try:
#         embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
#         return FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
#     except Exception as e:
#         logging.error(f"Failed to load existing FAISS DB: {e}")
#         st.error(f"Failed to load existing FAISS DB: {e}")
#         return None

# def get_chat_chain(vs):
#     prompt = ChatPromptTemplate.from_template("You are a table analysis expert.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:")
#     llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
#     return {"context": vs.as_retriever(), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# def clear_db():
#     if os.path.exists(DB_DIR):
#         shutil.rmtree(DB_DIR)
#         logging.info(f"FAISS DB directory '{DB_DIR}' cleared.")

# # --- Sidebar ---
# with st.sidebar:
#     st.image("img/ACL_Digital.png", width=180)
#     st.image("img/Cipla_Foundation.png", width=180)
#     st.markdown(""" <hr> """, unsafe_allow_html=True)
#     st.header("📂 Upload PDFs")
#     uploaded = st.file_uploader("Select PDFs", type="pdf", accept_multiple_files=True)
#     scanned_mode = st.checkbox("📸 PDF is scanned (image only)?")
#     run = st.button("📊 Extract & Index")

#     st.markdown(""" <hr> """, unsafe_allow_html=True)
#     st.header("🛠 Control")
#     if st.button("🗑 Clear DB"):
#         clear_db()
#         st.session_state.vs = None
#         st.success("DB cleared")
#     if st.button("🧹 Clear Chat"):
#         st.session_state.msgs = []
#         st.success("Chat cleared")

# # --- Main ---
# if "vs" not in st.session_state:
#     st.session_state.vs = load_existing_index()
# if "msgs" not in st.session_state:
#     st.session_state.msgs = []

# if run and uploaded:
#     st.session_state.msgs = []
#     with st.spinner("Processing documents and building index..."):
#         st.session_state.vs = load_and_index(uploaded, scanned_mode)
#     if st.session_state.vs:
#         st.session_state.msgs.append({"role": "assistant", "content": "Extraction & indexing done. Ask anything!"})

# for msg in st.session_state.msgs:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# if query := st.chat_input("Ask about the PDF content or tables..."):
#     st.session_state.msgs.append({"role": "user", "content": query})
#     with st.chat_message("user"):
#         st.markdown(query)

#     if st.session_state.vs:
#         chain = get_chat_chain(st.session_state.vs)
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 resp = "".join(chain.stream(query))
#                 st.markdown(resp)
#                 st.session_state.msgs.append({"role": "assistant", "content": resp})
#     else:
#         st.error("Please upload and process PDFs first to enable chat functionality.")












# --- ADVANCED PDF QA STREAMLIT APP WITH CHAT HISTORY, AGENTIC CHUNKING, FEEDBACK, DRILLDOWN ---

import os, tempfile, shutil, logging, json, re, uuid
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import pdfplumber, camelot, pytesseract
from pdf2image import convert_from_path
from io import StringIO
from difflib import SequenceMatcher
from datetime import datetime
import requests

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
FEEDBACK_LOG = "feedback_log.jsonl"
CHAT_LOG = "chat_sessions.jsonl"
CHAT_AUTOSAVE = "chat_autosave.json"

logging.basicConfig(level=logging.INFO, filename="app.log", format="%(asctime)s [%(levelname)s] %(message)s")

st.set_page_config(page_title="PDF QA with Tables", layout="wide")
st.title("📄 PDF QA + Intelligent Table Analysis")

# --- Utilities ---
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def stitch_tables(pages):
    stitched = []
    last_header = None
    for df in pages:
        header = ",".join(df.columns.tolist())
        if last_header and similar(header, last_header) > 0.8:
            stitched[-1] = pd.concat([stitched[-1], df], ignore_index=True)
        else:
            stitched.append(df)
        last_header = header
    return stitched

def save_chat_session(session_id, messages):
    with open(CHAT_LOG, "a") as f:
        f.write(json.dumps({"session_id": session_id, "timestamp": str(datetime.now()), "messages": messages}) + "\n")

def load_chat_sessions():
    if not os.path.exists(CHAT_LOG): return []
    with open(CHAT_LOG) as f:
        return [json.loads(line) for line in f.readlines()]

# --- OCR + LLM Table Extraction ---
def extract_scanned_pdf_with_ocr(pdf_path):
    try:
        images = convert_from_path(pdf_path)
        all_dfs, full_text = [], ""
        for img in images:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME).dropna()
            data = data[data.conf != '-1']
            data["line"] = (data.top.diff().abs() > 10).cumsum()
            grouped = data.groupby("line")
            rows = ["".join(g.sort_values("left")["text"]) for _, g in grouped]
            all_dfs.append(pd.DataFrame([r.split(",") for r in rows if len(r.split(",")) > 1]))
            full_text += pytesseract.image_to_string(img)
        tables = stitch_tables(all_dfs)
        all_csvs = "\n\n".join([df.to_csv(index=False) for df in tables])

        prompt = f"""You are a data expert. Clean and normalize the following tables:

{all_csvs}

- Fix headers
- Format numbers and dates
- Keep rows with meaningful data only
Return CSV only.
"""
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={"model": OLLAMA_LLM_MODEL, "prompt": prompt, "stream": False}, timeout=120)
        cleaned_csv = response.json().get("response", "")
        return cleaned_csv, full_text
    except Exception as e:
        logging.error(f"OCR+LLM failed: {e}")
        return "", ""

# --- Create Agentic Chunks ---
def create_agentic_chunks_from_csv(csv_text, file_name):
    chunks = []
    try:
        df = pd.read_csv(StringIO(csv_text))
        for i, row in df.iterrows():
            text = row.to_string()
            summary_prompt = f"Summarize the following table row for search indexing:\n{text}"
            resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={"model": OLLAMA_LLM_MODEL, "prompt": summary_prompt, "stream": False})
            summary = resp.json().get("response", text)
            doc = Document(
                page_content=f"{summary}\n\n[Drill Down: row_index={i}, file={file_name}]",
                metadata={"row_index": i, "source": file_name, "columns": list(row.index)}
            )
            chunks.append(doc)
    except Exception as e:
        logging.error(f"Agentic chunk creation failed: {e}")
    return chunks

# --- Load and Index Docs ---
@st.cache_resource(show_spinner=False)
def load_and_index(files, scanned_mode=False):
    all_chunks = []
    with tempfile.TemporaryDirectory() as td:
        for file in files:
            st.info(f"📄 Processing {file.name}...")
            path = os.path.join(td, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            loader = PyPDFLoader(path)
            pages = loader.load()
            if scanned_mode:
                csv_text, raw_text = extract_scanned_pdf_with_ocr(path)
                if not csv_text:
                    st.warning(f"⚠ OCR/LLM failed for {file.name}. Please check file quality.")
                chunks = create_agentic_chunks_from_csv(csv_text, file.name)
                all_chunks.extend(chunks)
            else:
                all_chunks.extend(pages)
    if not all_chunks:
        st.warning("No chunks created.")
        return None
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(all_chunks)
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(DB_DIR)
        st.success("✅ PDFs indexed and ready for Q&A!")
        return vs
    except Exception as e:
        st.error(f"Indexing failed: {e}")
        return None

# --- Feedback Loop ---
def save_feedback(query, answer):
    with open(FEEDBACK_LOG, "a") as f:
        f.write(json.dumps({"query": query, "answer": answer}) + "\n")

# --- Retrieval Chain ---
def get_chat_chain(vs):
    prompt = ChatPromptTemplate.from_template("You are a table analysis expert.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    return {"context": vs.as_retriever(), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# --- Sidebar UI ---
with st.sidebar:
    uploaded = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    scanned_mode = st.checkbox("Scanned PDF (image-based)?")
    start_indexing = st.button("Extract & Index")

    if st.button("Clear DB"):
        if os.path.exists(DB_DIR): shutil.rmtree(DB_DIR)
        st.session_state.vs = None
        st.success("DB cleared.")

    st.markdown("---")
    st.subheader("💬 Chat Sessions")
    sessions = load_chat_sessions()
    if sessions:
        for s in sessions[::-1][:5]:
            if st.button(f"Load chat {s['session_id'][:8]}"):
                st.session_state.msgs = s["messages"]
    if st.button("➕ Start New Chat"):
        if st.session_state.get("msgs"):
            save_chat_session(str(uuid.uuid4()), st.session_state.msgs)
        st.session_state.msgs = []

# --- Index Trigger ---
if start_indexing and uploaded:
    with st.spinner("Extracting and indexing..."):
        st.session_state.vs = load_and_index(uploaded, scanned_mode)

# --- Chat State Init ---
if "vs" not in st.session_state:
    st.session_state.vs = FAISS.load_local(DB_DIR, OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL), allow_dangerous_deserialization=True) if os.path.exists(DB_DIR) else None

if "msgs" not in st.session_state:
    if os.path.exists(CHAT_AUTOSAVE):
        with open(CHAT_AUTOSAVE) as f:
            st.session_state.msgs = json.load(f)
    else:
        st.session_state.msgs = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- Chat UI ---
for msg in st.session_state.msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask about the tables or text..."):
    st.chat_message("user").markdown(query)
    st.session_state.msgs.append({"role": "user", "content": query})
    if st.session_state.vs:
        chain = get_chat_chain(st.session_state.vs)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chain.invoke(query)
                st.markdown(answer)
                st.session_state.msgs.append({"role": "assistant", "content": answer})
                save_chat_session(st.session_state.session_id, st.session_state.msgs)
                with open(CHAT_AUTOSAVE, "w") as f:
                    json.dump(st.session_state.msgs, f)
                if st.button("Was this answer wrong? Click to flag."):
                    save_feedback(query, answer)
                    st.warning("Thanks! We'll review and improve future chunking.")
                drill_refs = re.findall(r"row_index=(\d+), file=(.*?)\]", answer)
                if drill_refs:
                    st.info(f"You can reprocess file {drill_refs[0][1]} and drill into row {drill_refs[0][0]} if needed.")
    else:
        st.error("Please upload and process PDFs first.")