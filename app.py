import streamlit as st
import os, tempfile, shutil
import pandas as pd
import numpy as np
from PIL import Image

# Ensure fitz is correctly imported from PyMuPDF
try:
    import fitz  # PyMuPDF
except ImportError:
    import sys
    sys.exit("\n❌ PyMuPDF (fitz) is not installed. Please run: pip install pymupdf\n")

import easyocr
import layoutparser as lp

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from table_extraction_with_llm import extract_tables_with_llm

# ========== Config ==========
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "mistral:7b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DB_DIR = "./faiss_db"

st.set_page_config(page_title="PDF Table Extractor and QA", layout="wide")
st.title("📄 Extract and Ask: Table Extraction from PDFs")

ocr_reader = easyocr.Reader(['en'], gpu=False)

# ========== Helper Functions ==========

def clean_and_format_df(df):
    df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
    df.fillna("", inplace=True)
    return df

def extract_tables_with_pdfplumber(pdf_path):
    import pdfplumber
    all_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if len(table) < 2:
                    continue
                header, *data = table
                try:
                    df = pd.DataFrame(data, columns=header)
                    df = clean_and_format_df(df)
                    all_tables.append(df)
                except:
                    continue
    return all_tables

def extract_tables_with_easyocr(pdf_path):
    all_tables = []
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ocr_result = ocr_reader.readtext(np.array(img), detail=0)
        lines = [line.strip() for line in ocr_result if line.strip()]
        rows = [line.split() for line in lines if len(line.split()) > 1]
        if rows and len(rows) > 1:
            try:
                df = pd.DataFrame(rows[1:], columns=rows[0])
                df = clean_and_format_df(df)
                all_tables.append(df)
            except:
                continue
    return all_tables

def extract_tables_with_layoutparser(pdf_path):
    from layoutparser import PaddleDetectionLayoutModel
    all_tables = []
    model = PaddleDetectionLayoutModel(
        config_path='lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config',
        threshold=0.5,
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        enforce_cpu=True
    )
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        layout = model.detect(np.array(img))
        for block in layout:
            if block.type != "Table":
                continue
            segment = block.crop_image(np.array(img))
            text = ocr_reader.readtext(segment, detail=0)
            lines = [line.strip() for line in text if line.strip()]
            rows = [line.split() for line in lines if len(line.split()) > 1]
            if rows and len(rows) > 1:
                try:
                    df = pd.DataFrame(rows[1:], columns=rows[0])
                    df = clean_and_format_df(df)
                    all_tables.append(df)
                except Exception:
                    continue
    return all_tables

def extract_tables(pdf_path):
    plumber_dfs = extract_tables_with_pdfplumber(pdf_path)
    ocr_dfs = extract_tables_with_easyocr(pdf_path)
    layout_dfs = extract_tables_with_layoutparser(pdf_path)
    llm_text = extract_tables_with_llm(pdf_path, model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL)
    all_dfs = plumber_dfs + ocr_dfs + layout_dfs
    result_strs = []
    for idx, df in enumerate(all_dfs):
        result_strs.append(f"Table {idx+1}:\n" + df.to_csv(index=False))
    result_strs.append("Tables Extracted via LLM:\n" + llm_text)
    return "\n\n".join(result_strs), all_dfs

@st.cache_resource(show_spinner=False)
def load_and_process_pdfs(uploaded_files):
    if not uploaded_files:
        return None
    with tempfile.TemporaryDirectory() as temp_dir:
        all_docs = []
        for file in uploaded_files:
            temp_path = os.path.join(temp_dir, file.name)
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            try:
                loader = PyPDFLoader(temp_path)
                pages = loader.load()
                all_docs.extend(pages)
                table_text, dfs = extract_tables(temp_path)
                all_docs.append(Document(page_content=table_text, metadata={"source": file.name}))
                if dfs:
                    excel_path = os.path.join("extracted_tables", f"{file.name}.xlsx")
                    os.makedirs("extracted_tables", exist_ok=True)
                    with pd.ExcelWriter(excel_path) as writer:
                        for i, df in enumerate(dfs):
                            df.to_excel(writer, sheet_name=f"Table_{i+1}", index=False)
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
                continue
        if not all_docs:
            st.error("No content loaded.")
            return None
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(all_docs)
        try:
            embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
            vs = FAISS.from_documents(chunks, embeddings)
            vs.save_local(DB_DIR)
            return vs
        except Exception as e:
            st.error(f"Embedding/vector DB error: {e}")
            return None

def load_existing_vector_store():
    if not os.path.exists(DB_DIR):
        return None
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        vs = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
        return vs
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

def get_rag_chain(vs):
    if vs is None:
        return None
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    prompt = ChatPromptTemplate.from_template(
        """You are an AI assistant analyzing complex tables. Use only the context provided.

Context:
{context}

Question: {question}

Answer:""")
    return ({"context": vs.as_retriever(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser())

def clear_database():
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

# Sidebar
with st.sidebar:
    st.image("img/Cipla_Foundation.png", width=150)
    st.image("img/ACL_Digital.png", width=150)
    st.header("📂 Upload PDF Files")
    uploaded = st.file_uploader("Select PDF files", type="pdf", accept_multiple_files=True)
    run = st.button("📊 Extract and Index")
    st.header("⚙️ Maintenance")
    if st.button("🗑️ Clear Vector Database"):
        clear_database()
        st.success("Vector database cleared")
    if st.button("🧹 Clear Chat History"):
        st.session_state.msgs = []
        st.success("Chat history cleared")

# Main App
if "vs" not in st.session_state:
    st.session_state.vs = load_existing_vector_store()
if "msgs" not in st.session_state:
    st.session_state.msgs = []

if run and uploaded:
    st.session_state.vs = load_and_process_pdfs(uploaded)
    if st.session_state.vs:
        st.session_state.msgs.append({"role": "assistant", "content": "✅ PDFs loaded and tables extracted. Ask your questions."})

for msg in st.session_state.msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask a question..."):
    st.session_state.msgs.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if st.session_state.vs:
        rag = get_rag_chain(st.session_state.vs)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ""
                for chunk in rag.stream(user_input):
                    answer += chunk
                st.markdown(answer)
                st.session_state.msgs.append({"role": "assistant", "content": answer})
    else:
        st.error("No vector store found. Please process PDFs first.")
