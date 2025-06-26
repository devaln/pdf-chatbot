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
    from pdf2image import convert_from_path
    from io import BytesIO
    import base64
    import pandas as pd
    import camelot
    import pdfplumber
    from tqdm import tqdm

    from mmocr.apis import TextRecInferencer

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.chat_models import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnablePassthrough

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_LLM_MODEL = "llama3:latest"
    OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
    DB_DIR = "./faiss_db"
    CHAT_DIR = "./chat_sessions"

    st.set_page_config(page_title="PDF QA with Tables", layout="wide")
    st.title("📄 PDF Extractor & QA (Scanned + Unscanned)")

    logging.basicConfig(level=logging.INFO, filename="app.log", format="%(asctime)s [%(levelname)s] %(message)s")

    def run_mmocr(images):
        inferencer = TextRecInferencer(rec='sar', det='dbnet', device='cpu')
        results = inferencer(images, return_vis=False)
        texts = []
        for res in results["predictions"]:
            txt = " ".join([r["text"] for r in res["instances"]])
            texts.append(txt)
        return "
".join(texts)

    def extract_text_with_llama(text):
        prompt = f"You are a table cleaning expert.

From the following OCR or raw text, extract all tables and present them in cleaned CSV format:

{text}"
        try:
            res = requests.post(
                url=f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": OLLAMA_LLM_MODEL, "prompt": prompt, "stream": False},
                timeout=180
            )
            return res.json().get("response", "")
        except Exception as e:
            logging.error(f"LLaMA extraction failed: {e}")
            return ""

    def extract_from_scanned(pdf_path):
        images = convert_from_path(pdf_path, dpi=300)
        texts = []
        for img in tqdm(images, desc="MMOCR"):
            texts.append(run_mmocr([img]))
        full_text = "
".join(texts)
        cleaned = extract_text_with_llama(full_text)
        return cleaned + "
" + full_text

    def extract_from_unscanned(pdf_path):
        text = ""
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    text += t + "
" if t else ""
                    page_tables = page.extract_tables()
                    for tbl in page_tables:
                        if tbl:
                            df = pd.DataFrame(tbl[1:], columns=tbl[0])
                            tables.append(df.to_csv(index=False))
        except Exception as e:
            logging.warning(f"pdfplumber failed: {e}")

        try:
            camelot_tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
            for table in camelot_tables:
                if table.df.shape[0] > 1:
                    tables.append(table.df.to_csv(index=False, header=False))
        except Exception as e:
            logging.warning(f"camelot failed: {e}")

        raw = text + "
" + "
".join(tables)
        cleaned = extract_text_with_llama(raw)
        return cleaned + "
" + raw

    @st.cache_resource(show_spinner=False)
    def load_and_index(files, scanned_mode):
        all_docs = []
        with tempfile.TemporaryDirectory() as td:
            for file in files:
                path = os.path.join(td, file.name)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                try:
                    if scanned_mode:
                        final_text = extract_from_scanned(path)
                    else:
                        final_text = extract_from_unscanned(path)
                    all_docs.append(Document(page_content=final_text, metadata={"source": file.name}))
                except Exception as e:
                    st.error(f"Failed: {file.name} – {e}")

        if not all_docs:
            return None

        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(all_docs)
        try:
            embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
            vs = FAISS.from_documents(chunks, embeddings)
            vs.save_local(DB_DIR)
            return vs
        except Exception as e:
            st.error(f"FAISS error: {e}")
            return None

    def get_chat_chain(vs):
        prompt = ChatPromptTemplate.from_template("You are a helpful assistant.

Context:
{context}

Question: {question}

Answer:")
        llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL)
        return {"context": vs.as_retriever(), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

    def clear_db():
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)

    # Sidebar
    with st.sidebar:
        st.header("📂 Upload PDFs")
        uploaded = st.file_uploader("Choose PDFs", type="pdf", accept_multiple_files=True)
        scanned_mode = st.checkbox("📷 Scanned PDF?")
        run = st.button("📊 Extract & Index")
        if st.button("🗑 Clear DB"):
            clear_db()
            st.session_state.vs = None
        if st.button("🔁 Clear Chat"):
            st.session_state.msgs = []

    # Main Chat UI
    if "vs" not in st.session_state:
        if os.path.exists(DB_DIR):
            st.session_state.vs = FAISS.load_local(DB_DIR, OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL), allow_dangerous_deserialization=True)
    if "msgs" not in st.session_state:
        st.session_state.msgs = []

    if run and uploaded:
        st.session_state.msgs = []
        with st.spinner("Processing PDFs..."):
            st.session_state.vs = load_and_index(uploaded, scanned_mode)
        if st.session_state.vs:
            st.session_state.msgs.append({"role": "assistant", "content": "Extraction & indexing done! Ask your question."})

    for msg in st.session_state.msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask about the PDFs..."):
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
            st.error("Please upload and index PDF files first.")