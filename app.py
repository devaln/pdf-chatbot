import os
import shutil
import fitz  # PyMuPDF
import tempfile
import streamlit as st
import pandas as pd

from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import layoutparser as lp
import easyocr
from PIL import Image
from io import BytesIO

# === Constants ===
DB_DIR = "vectorstore"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral:7b-instruct"
OCR_MODEL = easyocr.Reader(['en'])

# === Helper: Extract text from PDFs ===
def extract_text_from_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# === Helper: Extract and clean tables from scanned/digital PDFs ===
def extract_tables_with_ocr(pdf_file):
    tables = []
    doc = fitz.open(pdf_file)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        image = Image.open(BytesIO(pix.tobytes()))
        layout = lp.PaddleDetectionLayoutModel('lp://TableBank/faster_rcnn_R_50_FPN_3x/config',
                                               extra_config={"box_threshold": 0.5})
        layout_result = layout.detect(image)

        for block in layout_result:
            if block.type == "Table":
                x1, y1, x2, y2 = map(int, block.block.bounding_box)
                cropped_image = image.crop((x1, y1, x2, y2))
                result = OCR_MODEL.readtext(np.array(cropped_image), detail=0)
                if result:
                    table_df = pd.DataFrame([r.split() for r in result if r.strip()])
                    tables.append(table_df)
    return tables

# === Helper: Clean tables with LLaMA 3 ===
def clean_table_with_llm(table_df):
    llm = Ollama(model="llama3", base_url=OLLAMA_BASE_URL)
    prompt = f"Clean and normalize the following table:\n\n{table_df.to_csv(index=False)}"
    cleaned = llm(prompt)
    try:
        return pd.read_csv(BytesIO(cleaned.encode()), on_bad_lines='skip')
    except Exception:
        return table_df

# === Helper: Build index for RAG ===
def build_index(pdf_files):
    all_docs = []
    for pdf in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.read())
            tmp_path = tmp.name
        docs = extract_text_from_pdf(tmp_path)
        all_docs.extend(docs)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    vectorstore.save_local(DB_DIR)
    return vectorstore

# === Helper: QA Chain ===
def get_chain(vectorstore):
    llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# === Streamlit App ===
st.set_page_config(page_title="PDF Analyzer", layout="wide")

# === Sidebar ===
with st.sidebar:
    st.image("img/ACL_Digital.png", width=160)
    st.image("img/Cipla_Foundation.png", width=160)
    st.markdown("---")
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")

    if st.button("Index") and uploaded_files:
        with st.spinner("Indexing documents... This might take a moment"):
            st.session_state.vs = build_index(uploaded_files)
        st.session_state.chat = [{"role": "assistant", "content": "Indexing done! You can now ask questions."}]

    if st.button("Clear"):
        shutil.rmtree(DB_DIR, ignore_errors=True)
        st.session_state.chat = []
        st.session_state.vs = None

# === Chat Session State ===
if "vs" not in st.session_state:
    st.session_state.vs = FAISS.load_local(DB_DIR, OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)) if os.path.exists(DB_DIR) else None

if "chat" not in st.session_state:
    st.session_state.chat = []

# === Display Chat ===
for msg in st.session_state.chat:
    st.chat_message(msg["role"]).markdown(msg["content"])

# === Query Input ===
if query := st.chat_input("Ask a question about your PDFs..."):
    st.session_state.chat.append({"role": "user", "content": query})
    if st.session_state.vs:
        chain = get_chain(st.session_state.vs)
        with st.spinner("Thinking..."):
            response = "".join(chain.stream(query))
        st.session_state.chat.append({"role": "assistant", "content": response})
        st.chat_message("assistant").markdown(response)
    else:
        st.chat_message("assistant").markdown("Please upload and index PDFs first.")

# === Optional: Display Extracted Tables ===
if uploaded_files:
    st.markdown("### Extracted Tables")
    for pdf in uploaded_files:
        st.subheader(f"Tables from {pdf.name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.read())
            tmp_path = tmp.name
        tables = extract_tables_with_ocr(tmp_path)
        for i, table_df in enumerate(tables):
            st.markdown(f"*Table {i+1} (raw OCR):*")
            st.dataframe(table_df)
            cleaned_df = clean_table_with_llm(table_df)
            st.markdown(f"*Table {i+1} (cleaned with LLaMA 3):*")
            st.dataframe(cleaned_df)
