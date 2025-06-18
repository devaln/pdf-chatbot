import streamlit as st
import os, tempfile, shutil
import pandas as pd
import numpy as np
from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:
    import sys
    sys.exit("❌ PyMuPDF not found. Run: pip install pymupdf")

import easyocr
import layoutparser as lp

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from table_extraction_with_llm import extract_tables_with_llm

# == CONFIG ==
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "mistral:7b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DB_DIR = "./faiss_db"

st.set_page_config(page_title="PDF Table Extractor & QA", layout="wide")
st.title("📄 Extract & Ask: Table Extraction from PDFs")

ocr_reader = easyocr.Reader(['en'], gpu=False)

def clean_and_format_df(df):
    df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
    return df.fillna("").reset_index(drop=True)

def extract_tables_with_pdfplumber(path):
    import pdfplumber
    dfs = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                if len(table) < 2: continue
                df = pd.DataFrame(table[1:], columns=table[0])
                dfs.append(clean_and_format_df(df))
    return dfs

def extract_tables_with_easyocr(path):
    dfs = []
    doc = fitz.open(path)
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        lines = [l.strip() for l in easyocr.Reader(['en'], gpu=False).readtext(np.array(img), detail=0) if l.strip()]
        rows = [l.split() for l in lines if len(l.split()) > 1]
        if len(rows) > 1:
            try:
                df = pd.DataFrame(rows[1:], columns=rows[0])
                dfs.append(clean_and_format_df(df))
            except: pass
    return dfs

def extract_tables_with_layoutparser(path):
    from layoutparser import PaddleDetectionLayoutModel
    model = PaddleDetectionLayoutModel("lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config", enforce_cpu=True)
    dfs = []
    doc = fitz.open(path)
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        layout = model.detect(np.array(img))
        for block in layout:
            if block.type != "Table": continue
            segment = block.crop_image(np.array(img))
            lines = [l.strip() for l in ocr_reader.readtext(segment, detail=0) if l.strip()]
            rows = [l.split() for l in lines if len(l.split()) > 1]
            if len(rows) > 1:
                try:
                    df = pd.DataFrame(rows[1:], columns=rows[0])
                    dfs.append(clean_and_format_df(df))
                except: pass
    return dfs

def extract_tables(path):
    dfs = extract_tables_with_pdfplumber(path)
    dfs += extract_tables_with_easyocr(path)
    dfs += extract_tables_with_layoutparser(path)
    llm_csv = extract_tables_with_llm(path, model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL)
    
    text = "\n\n".join(f"Table {i+1}:\n{df.to_csv(index=False)}" for i, df in enumerate(dfs))
    text += "\n\nTables Extracted via LLM:\n" + llm_csv
    return text, dfs

@st.cache_resource(show_spinner=False)
def load_and_process_pdfs(files):
    os.makedirs("extracted_tables", exist_ok=True)
    docs = []
    for file in files:
        fpath = os.path.join(tempfile.gettempdir(), file.name)
        with open(fpath, 'wb') as f: f.write(file.getbuffer())
        try:
            pages = PyPDFLoader(fpath).load()
            docs.extend(pages)
            text, dfs = extract_tables(fpath)
            docs.append(Document(page_content=text, metadata={"source": file.name}))
            if dfs:
                writer = pd.ExcelWriter(f"extracted_tables/{file.name}.xlsx", engine="openpyxl")
                for i, df in enumerate(dfs):
                    df.to_excel(writer, sheet_name=f"Table_{i+1}", index=False)
                writer.save()
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")
    if not docs: return None
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    vs = FAISS.from_documents(chunks, OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL))
    vs.save_local(DB_DIR)
    return vs

def load_existing_vector_store():
    if not os.path.exists(DB_DIR): return None
    return FAISS.load_local(DB_DIR, OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL), allow_dangerous_deserialization=True)

def get_rag_chain(vs):
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL)
    prompt = ChatPromptTemplate.from_template(
        "You are an AI assistant answering questions using only the provided CONTEXT.\n\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return {"context": vs.as_retriever(), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

def clear_db():
    if os.path.exists(DB_DIR): shutil.rmtree(DB_DIR)

# --- UI Setup ---
with st.sidebar:
    st.image("img/Cipla_Foundation.png", width=150)
    st.image("img/ACL_Digital.png", width=150)
    st.header("Upload PDF Files")
    uploaded = st.file_uploader("Select PDF(s)", type="pdf", accept_multiple_files=True)
    run = st.button("Extract & Index")
    st.header("Maintenance")
    if st.button("Clear Vector DB"): clear_db(); st.success("Vector DB cleared")
    if st.button("Clear Chat History"): st.session_state.msgs = []; st.success("Chat cleared")

if "vs" not in st.session_state: st.session_state.vs = load_existing_vector_store()
if "msgs" not in st.session_state: st.session_state.msgs = []

if run and uploaded:
    st.session_state.vs = load_and_process_pdfs(uploaded)
    if st.session_state.vs:
        st.session_state.msgs.append({"role": "assistant", "content": "✅ PDFs processed! Ask your questions."})

for msg in st.session_state.msgs:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if user_input := st.chat_input("Ask a question..."):
    st.session_state.msgs.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    if st.session_state.vs:
        chain = get_rag_chain(st.session_state.vs)
        with st.chat_message("assistant"):
            with st.spinner("🔍 Thinking..."):
                resp = chain.invoke({"question": user_input})
                st.markdown(resp)
                st.session_state.msgs.append({"role": "assistant", "content": resp})
    else:
        st.error("Please upload and index PDFs first.")
