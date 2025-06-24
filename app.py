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
# import json
# import uuid
# from datetime import datetime

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
# CHAT_DIR = "./chat_sessions"

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

#         llm_prompt = f"""You are a table understanding expert. Extract all tables from the following OCR text and convert them to CSV format:

# {full_text}

# Only return CSV-formatted tables."""

#         response = requests.post(
#             url=f"{OLLAMA_BASE_URL}/api/generate",
#             json={"model": OLLAMA_LLM_MODEL, "prompt": llm_prompt, "stream": False},
#             timeout=120
#         )
#         result = response.json()
#         csv_text = result.get("response", "")
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
#                 loader = PyPDFLoader(path)
#                 all_docs.extend(loader.load())
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
#     st.markdown("""<hr>""", unsafe_allow_html=True)

#     st.header("📂 Upload PDFs")
#     uploaded = st.file_uploader("Select PDFs", type="pdf", accept_multiple_files=True)
#     scanned_mode = st.checkbox("📸 PDF is scanned (image only)?")
#     run = st.button("📊 Extract & Index")

#     st.markdown("""<hr>""", unsafe_allow_html=True)
#     st.header("🛠 Control")
#     if st.button("🗑 Clear DB"):
#         clear_db()
#         st.session_state.vs = None
#         st.success("DB cleared")
#     if st.button("🧹 Clear Chat"):
#         st.session_state.msgs = []
#         st.success("Chat cleared")

#     st.markdown("""<hr>""", unsafe_allow_html=True)
#     st.header("💬 Chat History")
#     os.makedirs(CHAT_DIR, exist_ok=True)

#     def summarize_chat(msgs):
#         for msg in msgs:
#             if msg["role"] == "user" and msg["content"].strip():
#                 first_line = msg["content"].strip().split("\n")[0]
#                 summary = first_line.strip()[:40].replace(" ", "_").replace("?", "").replace(":", "")
#                 return summary.lower()
#         return f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

#     if "chat_id" not in st.session_state:
#         base = summarize_chat(st.session_state.get("msgs", []))
#         st.session_state.chat_id = f"{base}_{uuid.uuid4().hex[:4]}"
#         with open(os.path.join(CHAT_DIR, f"{st.session_state.chat_id}.json"), "w") as f:
#             json.dump([], f)

#     if st.button("➕ New Chat"):
#         base = summarize_chat(st.session_state.get("msgs", []))
#         st.session_state.chat_id = f"{base}_{uuid.uuid4().hex[:4]}"
#         st.session_state.msgs = []
#         with open(os.path.join(CHAT_DIR, f"{st.session_state.chat_id}.json"), "w") as f:
#             json.dump([], f)
#         st.rerun()

#     session_files = sorted(
#         [f for f in os.listdir(CHAT_DIR) if f.endswith(".json")],
#         key=lambda x: os.path.getmtime(os.path.join(CHAT_DIR, x)),
#         reverse=True
#     )[:10]

#     for fname in session_files:
#         label = fname.replace(".json", "").replace("_", " ").title()
#         if st.button(f"💬 {label}"):
#             st.session_state.chat_id = fname.replace(".json", "")
#             with open(os.path.join(CHAT_DIR, fname), "r") as f:
#                 st.session_state.msgs = json.load(f)
#             st.session_state.vs = load_existing_index()  # ✅ Fix: Reload vector index
#             st.rerun()

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

# # --- Save Chat ---
# if "chat_id" in st.session_state:
#     with open(os.path.join(CHAT_DIR, f"{st.session_state.chat_id}.json"), "w") as f:
#         json.dump(st.session_state.msgs, f)












# (all your existing imports remain unchanged)
import os
import tempfile
import shutil
import logging
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import pdfplumber
import camelot
import torch
import requests
import pytesseract
from pdf2image import convert_from_path
import json
import uuid
import re  # ✅ NEW for document ID cleanup
from datetime import datetime

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
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            background-color: white !important;
            border-right: 2px solid #e0e0e0 !important;
        }
        .chat-history-button button {
            text-align: left !important;
            font-size: 0.85rem !important;
            width: 100% !important;
            margin-bottom: 4px;
            background-color: #f8f9fa !important;
            border: 1px solid #d1d1d1 !important;
        }
    </style>
""", unsafe_allow_html=True)
st.title("\U0001F4C4 PDF Text & Table Extractor + Chat QA")

# --- (existing helper functions and extract logic remain unchanged) ---
# --- keep all your extract_tables_pdfplumber, load_and_index, etc. unchanged ---

# --- Sidebar ---
with st.sidebar:
    st.image("img/ACL_Digital.png", width=180)
    st.image("img/Cipla_Foundation.png", width=180)
    st.markdown("""<hr>""", unsafe_allow_html=True)

    st.header("📂 Upload PDFs")
    uploaded = st.file_uploader("Select PDFs", type="pdf", accept_multiple_files=True)
    scanned_mode = st.checkbox("📸 PDF is scanned (image only)?")
    run = st.button("📊 Extract & Index")

    st.markdown("""<hr>""", unsafe_allow_html=True)
    st.header("🛠 Control")
    if st.button("🗑 Clear DB"):
        clear_db() # type: ignore
        st.session_state.vs = None
        st.success("DB cleared")
    if st.button("🧹 Clear Chat"):
        st.session_state.msgs = []
        st.success("Chat cleared")

    if st.button("🧨 Delete All Chat History"):
        for f in os.listdir(CHAT_DIR):
            if f.endswith(".json"):
                os.remove(os.path.join(CHAT_DIR, f))
        st.success("All chat history deleted!")

    st.markdown("""<hr>""", unsafe_allow_html=True)
    st.header("💬 Chat History")
    os.makedirs(CHAT_DIR, exist_ok=True)

    def summarize_chat(msgs):
        for msg in msgs:
            if msg["role"] == "user" and msg["content"].strip():
                first_line = msg["content"].strip().split("\n")[0]
                summary = first_line.strip()[:40].replace(" ", "_").replace("?", "").replace(":", "")
                return summary.lower()
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

    session_files = sorted(
        [f for f in os.listdir(CHAT_DIR) if f.endswith(".json")],
        key=lambda x: os.path.getmtime(os.path.join(CHAT_DIR, x)),
        reverse=True
    )[:10]

    for fname in session_files:
        label = fname.replace(".json", "").replace("_", " ").title()
        with st.container():
            with st.container():
                if st.button(f"💬 {label}", key=f"chat_{fname}", use_container_width=True):
                    st.session_state.chat_id = fname.replace(".json", "")
                    with open(os.path.join(CHAT_DIR, fname), "r") as f:
                        st.session_state.msgs = json.load(f)
                    st.session_state.vs = load_existing_index() # type: ignore
                    st.rerun()

# --- Main ---
if "vs" not in st.session_state:
    st.session_state.vs = load_existing_index() # type: ignore
if "msgs" not in st.session_state:
    st.session_state.msgs = []

if run and uploaded:
    st.session_state.msgs = []
    with st.spinner("Processing documents and building index..."):
        st.session_state.vs = load_and_index(uploaded, scanned_mode) # type: ignore
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
        chain = get_chat_chain(st.session_state.vs) # type: ignore
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                raw_resp = "".join(chain.stream(query))
                # ✅ Strip Document(id='...') patterns
                cleaned_resp = re.sub(r"Document\(id='[^']+'\)", "", raw_resp).strip()
                st.markdown(cleaned_resp)
                st.session_state.msgs.append({"role": "assistant", "content": cleaned_resp})
    else:
        st.error("Please upload and process PDFs first to enable chat functionality.")

# --- Save Chat ---
if "chat_id" in st.session_state:
    with open(os.path.join(CHAT_DIR, f"{st.session_state.chat_id}.json"), "w") as f:
        json.dump(st.session_state.msgs, f)