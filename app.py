# app.py (OCR-only, no Donut)

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

logging.basicConfig(level=logging.INFO, filename="app.log", format="%(asctime)s [%(levelname)s] %(message)s")

st.set_page_config(page_title="PDF QA with Tables", layout="wide")
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            background-color: white !important;
            border-right: 2px solid #e0e0e0 !important;
        }
    </style>
""", unsafe_allow_html=True)
st.title("\U0001F4C4 PDF Text & Table Extractor + Chat QA")

# --- Helpers ---
def clean_df(df):
    df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
    return df.fillna("")

def extract_tables_pdfplumber(pdf_path):
    dfs = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tbls = page.extract_tables()
                for table in tbls:
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        dfs.append(clean_df(df))
    except Exception as e:
        logging.warning(f"pdfplumber failed for {os.path.basename(pdf_path)}: {e}")
    return dfs

def extract_tables_camelot(pdf_path):
    dfs = []
    for flavor in ["lattice", "stream"]:
        try:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor=flavor)
            for t in tables:
                df = t.df
                if df.shape[0] > 1 and df.shape[1] > 1:
                    dfs.append(clean_df(df))
        except Exception as e:
            logging.warning(f"camelot {flavor} failed for {os.path.basename(pdf_path)}: {e}")
    return dfs

def extract_scanned_pdf_with_ocr(pdf_path):
    try:
        images = convert_from_path(pdf_path)
        full_text = ""
        for img in images:
            text = pytesseract.image_to_string(img)
            full_text += text + "\n"

        llm_prompt = f"""You are a table understanding expert.
Extract all tables from the following OCR text and convert them to CSV format:

{full_text}

Only return CSV-formatted tables."""

        response = requests.post(
            url=f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_LLM_MODEL, "prompt": llm_prompt, "stream": False},
            timeout=120
        )
        result = response.json()
        csv_text = result.get("response", "")

        st.subheader("LLM‑Structured Tables from OCR")
        st.text(csv_text)
        return csv_text, full_text
    except Exception as e:
        logging.error(f"OCR + LLM extraction failed: {e}")
        st.error(f"OCR + LLM failed: {e}")
        return "", ""

def extract_all_tables(pdf_path, scanned_mode=False):
    if scanned_mode:
        return extract_scanned_pdf_with_ocr(pdf_path)

    dfs = extract_tables_pdfplumber(pdf_path)
    dfs += extract_tables_camelot(pdf_path)

    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
    except Exception as e:
        logging.error(f"PDF text extraction failed: {e}")
        text = ""

    prompt = f"You are a table understanding expert.\n\nExtract all tables from the following document and convert them to CSV format:\n\n{text}\n\nOnly return CSV-formatted tables."

    try:
        response = requests.post(
            url=f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=120
        )
        result = response.json()
        llm_csv = result.get("response", "")
    except Exception as e:
        logging.error(f"LLM extraction failed: {e}")
        llm_csv = ""

    table_texts = []
    for i, df in enumerate(dfs):
        st.subheader(f"Table {i+1} (Raw)")
        st.dataframe(df)
        table_texts.append(f"Table {i+1}:\n{df.to_csv(index=False)}")

    st.subheader("LLM‑Structured Tables")
    st.text(llm_csv)
    table_texts.append("LLM-Structured Tables:\n" + llm_csv)

    return "\n\n".join(table_texts), text

@st.cache_resource(show_spinner=False)
def load_and_index(files, scanned_mode=False):
    all_docs = []
    with tempfile.TemporaryDirectory() as td:
        for file in files:
            path = os.path.join(td, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            try:
                # st.info(f"📄 Loading {file.name}...")
                loader = PyPDFLoader(path)
                all_docs.extend(loader.load())
                # st.info("🔍 Extracting tables...")
                text_csv, raw_text = extract_all_tables(path, scanned_mode)
                all_docs.append(Document(page_content=text_csv + "\n" + raw_text, metadata={"source": file.name}))
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

# --- Sidebar ---
with st.sidebar:
    st.image("img/ACL_Digital.png", width=180)
    st.image("img/Cipla_Foundation.png", width=180)
    st.markdown(""" <hr> """, unsafe_allow_html=True)
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

# --- Main ---
if "vs" not in st.session_state:
    st.session_state.vs = load_existing_index()
if "msgs" not in st.session_state:
    st.session_state.msgs = []

if run and uploaded:
    st.session_state.msgs = []
    with st.spinner("Processing documents and building index..."):
        st.session_state.vs = load_and_index(uploaded, scanned_mode)
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



# import os
# # Set the environment variable to allow multiple OpenMP runtimes.
# # This should be done at the very beginning of the script to be effective.
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# import tempfile
# import shutil
# import logging

# import streamlit as st
# import pandas as pd
# import numpy as np
# from PIL import Image
# import fitz  # PyMuPDF
# import easyocr
# import pdfplumber
# import camelot

# # Import the custom table extraction function
# from table_extraction_with_llm import extract_tables_with_llm

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.chat_models import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.documents import Document
# from langchain_core.runnables import RunnablePassthrough

# # --- Configuration Constants ---
# OLLAMA_BASE_URL = "http://localhost:11434"
# OLLAMA_LLM_MODEL = "llama3:latest"
# OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
# DB_DIR = "./faiss_db" # Directory to store FAISS vector store

# # --- Logging Configuration ---
# # Set up basic logging to a file for debugging
# logging.basicConfig(level=logging.INFO, filename="app.log",
#                     format="%(asctime)s [%(levelname)s] %(message)s")

# # --- Streamlit Page Configuration ---
# st.set_page_config(page_title="PDF Table & Text Extractor", layout="wide")
# # Set sidebar background color to white
# st.markdown("""
#     <style>
#         section[data-testid="stSidebar"] {
#             background-color: white !important;
#             border-right: 2px solid #e0e0e0 !important;
#         }
#     </style>
# """, unsafe_allow_html=True)
# st.title("📄 PDF Text & Table Extractor + Chat QA")

# # Initialize EasyOCR reader once for performance
# ocr_reader = easyocr.Reader(['en'], gpu=False)

# # --- Helper Functions ---

# def clean_df(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Cleans a DataFrame by deduping column names and filling NaN values with empty strings.
#     Args:
#         df (pd.DataFrame): The input DataFrame.
#     Returns:
#         pd.DataFrame: The cleaned DataFrame.
#     """
#     # Deduplicate column names (e.g., if multiple columns have the same header)
#     df.columns = pd.io.parsers.ParserBase({'names': df.columns})\
#                      ._maybe_dedup_names(df.columns)
#     return df.fillna("") # Fill any NaN values with empty strings for cleaner output


# def extract_tables_pdfplumber(pdf_path: str) -> list[pd.DataFrame]:
#     """
#     Extracts tables from a PDF using pdfplumber.
#     Args:
#         pdf_path (str): Path to the PDF file.
#     Returns:
#         list[pd.DataFrame]: A list of extracted tables as pandas DataFrames.
#     """
#     dfs = []
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             for page in pdf.pages:
#                 tbls = page.extract_tables()
#                 for table in tbls:
#                     # The first row is usually the header, rest are data
#                     if table: # Ensure table is not empty
#                         df = pd.DataFrame(table[1:], columns=table[0])
#                         dfs.append(clean_df(df))
#     except Exception as e:
#         # Log a warning if pdfplumber fails to extract tables
#         logging.warning(f"pdfplumber failed for {os.path.basename(pdf_path)}: {e}")
#     return dfs


# def extract_tables_camelot(pdf_path: str) -> list[pd.DataFrame]:
#     """
#     Extracts tables from a PDF using Camelot (lattice and stream flavors).
#     Args:
#         pdf_path (str): Path to the PDF file.
#     Returns:
#         list[pd.DataFrame]: A list of extracted tables as pandas DataFrames.
#     """
#     dfs = []
#     for flavor in ["lattice", "stream"]: # Try both table extraction methods
#         try:
#             tables = camelot.read_pdf(pdf_path, pages='all', flavor=flavor)
#             for t in tables:
#                 df = t.df
#                 # Add DataFrame only if it has more than one row and column
#                 if df.shape[0] > 1 and df.shape[1] > 1:
#                     dfs.append(clean_df(df))
#         except Exception as e:
#             # Log a warning if camelot fails for a specific flavor
#             logging.warning(f"camelot {flavor} failed for {os.path.basename(pdf_path)}: {e}")
#     return dfs


# def extract_easyocr_tables(pdf_path: str) -> list[pd.DataFrame]:
#     """
#     Extracts tables from a PDF using EasyOCR by processing page images.
#     This method is more robust for image-based tables.
#     Args:
#         pdf_path (str): Path to the PDF file.
#     Returns:
#         list[pd.DataFrame]: A list of extracted tables as pandas DataFrames.
#     """
#     tables = []
#     try:
#         doc = fitz.open(pdf_path)
#     except Exception as e:
#         logging.error(f"Error opening PDF {pdf_path}: {e}")
#         return tables

#     for page in doc:
#         try:
#             # Get a pixmap (image) of the page
#             pix = page.get_pixmap()
#             # Convert pixmap to PIL Image
#             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#             # Perform OCR on the image to get text lines
#             lines = [l.strip() for l in ocr_reader.readtext(np.array(img), detail=0) if l.strip()]
#             # Attempt to structure lines into rows based on spaces
#             rows = [line.split() for line in lines if len(line.split()) > 1]
#             if len(rows) > 1: # Ensure there's at least a header and one row of data
#                 df = pd.DataFrame(rows[1:], columns=rows[0])
#                 tables.append(clean_df(df))
#         except Exception:
#             # Continue to the next page if OCR extraction fails for the current page
#             continue
#     return tables


# def extract_all_tables(pdf_path: str) -> tuple[str, list[pd.DataFrame]]:
#     """
#     Extracts tables using multiple methods (pdfplumber, camelot, easyocr)
#     and then uses an LLM to structure additional tables.
#     Args:
#         pdf_path (str): Path to the PDF file.
#     Returns:
#         tuple[str, list[pd.DataFrame]]: A tuple containing:
#             - A combined string of all extracted table data (in CSV format).
#             - A list of pandas DataFrames from rule-based and OCR methods.
#     """
#     dfs = []
#     # Combine tables from different extraction methods
#     dfs += extract_tables_pdfplumber(pdf_path)
#     dfs += extract_tables_camelot(pdf_path)
#     dfs += extract_easyocr_tables(pdf_path)

#     # Use LLM for potentially more complex or unstructured table extraction
#     llm_csv = extract_tables_with_llm(pdf_path,
#                                       model=OLLAMA_LLM_MODEL,
#                                       base_url=OLLAMA_BASE_URL)
#     # Provide feedback if LLM extraction was not successful
#     if "No extractable content" in llm_csv or "LLM extraction failed" in llm_csv:
#         st.warning(f"⚠ LLM couldn't structure tables in {os.path.basename(pdf_path)}")

#     table_texts = []
#     # Display and convert rule-based/OCR extracted tables to CSV for RAG
#     for i, df in enumerate(dfs):
#         st.subheader(f"Table {i+1} (Raw)")
#         st.dataframe(df) # Display raw DataFrame in Streamlit
#         csv = df.to_csv(index=False)
#         table_texts.append(f"Table {i+1}:\n{csv}")

#     # Display LLM-structured tables and add to RAG context
#     st.subheader("LLM‑Structured Tables")
#     st.text(llm_csv) # Display the raw CSV string from LLM
#     table_texts.append("LLM-Structured Tables:\n" + llm_csv)

#     return "\n\n".join(table_texts), dfs


# @st.cache_resource(show_spinner=False)
# def load_and_index(files):
#     """
#     Loads PDF documents, extracts text and tables, chunks them, and creates/updates a FAISS vector store.
#     Uses Streamlit's cache_resource to prevent re-running on every interaction.
#     Args:
#         files (list): List of uploaded Streamlit file objects.
#     Returns:
#         FAISS: The FAISS vector store, or None if no documents were processed.
#     """
#     all_docs = []
#     # Use a temporary directory to save uploaded PDF files
#     with tempfile.TemporaryDirectory() as td:
#         for file in files:
#             path = os.path.join(td, file.name)
#             with open(path, "wb") as f:
#                 f.write(file.getbuffer()) # Write uploaded file content to temp file
#             try:
#                 # Load PDF content using PyPDFLoader
#                 loader = PyPDFLoader(path)
#                 all_docs.extend(loader.load())

#                 # Extract tables and their text representations
#                 text_csv, _ = extract_all_tables(path)
#                 # Add table data as a separate document for RAG
#                 all_docs.append(Document(page_content=text_csv,
#                                          metadata={"source": file.name}))
#             except Exception as e:
#                 # Log and display errors for failed PDF processing
#                 logging.error(f"Failed to process {file.name}: {e}")
#                 st.error(f" Failed to process {file.name}: {e}")

#     if not all_docs:
#         st.warning("No documents were successfully loaded or extracted.")
#         return None

#     # Chunk the documents for better retrieval
#     chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)\
#         .split_documents(all_docs)
#     try:
#         # Initialize Ollama embeddings and create/update FAISS index
#         embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL,
#                                       base_url=OLLAMA_BASE_URL)
#         vs = FAISS.from_documents(chunks, embeddings)
#         vs.save_local(DB_DIR) # Save the FAISS index locally
#         st.success("Documents processed and indexed successfully!")
#         return vs
#     except Exception as e:
#         logging.error(f"FAISS indexing error: {e}")
#         st.error(f"FAISS indexing error: {e}")
#         return None


# def load_existing_index():
#     """
#     Loads an existing FAISS vector store from the local directory.
#     Returns:
#         FAISS: The loaded FAISS vector store, or None if not found or an error occurs.
#     """
#     if not os.path.exists(DB_DIR):
#         return None # Return None if the FAISS directory doesn't exist
#     try:
#         embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL,
#                                       base_url=OLLAMA_BASE_URL)
#         # Load the local FAISS index, allowing dangerous deserialization for simplicity
#         return FAISS.load_local(DB_DIR, embeddings,
#                                 allow_dangerous_deserialization=True)
#     except Exception as e:
#         logging.error(f"Failed to load existing FAISS DB: {e}")
#         st.error(f"Failed to load existing FAISS DB: {e}")
#         return None


# def get_chat_chain(vs):
#     """
#     Creates and returns a LangChain RAG (Retrieval Augmented Generation) chain.
#     Args:
#         vs (FAISS): The FAISS vector store for retrieval.
#     Returns:
#         Runnable: A LangChain runnable chain for chat QA.
#     """
#     # Define the prompt template for the LLM
#     prompt = ChatPromptTemplate.from_template(
#         "You are a table analysis expert.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
#     )
#     # Initialize the ChatOllama LLM
#     llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
#     # Construct the RAG chain: retrieve context -> pass to prompt -> LLM -> parse output
#     return {"context": vs.as_retriever(), "question": RunnablePassthrough()} | \
#            prompt | llm | StrOutputParser()


# def clear_db():
#     """Clears the local FAISS database directory."""
#     if os.path.exists(DB_DIR):
#         shutil.rmtree(DB_DIR)
#         logging.info(f"FAISS DB directory '{DB_DIR}' cleared.")


# # --- Streamlit Sidebar ---
# with st.sidebar:
#     st.image("img/ACL_Digital.png", width=180)
#     st.image("img/Cipla_Foundation.png", width=180)
#     st.markdown(""" <hr> """, unsafe_allow_html=True)
#     st.header("📂 Upload PDFs")
#     # File uploader widget for PDF files
#     uploaded = st.file_uploader("Select PDFs (multi-page OK)", type="pdf", accept_multiple_files=True)
#     run = st.button("📊 Extract & Index")

#     st.markdown(""" <hr> """, unsafe_allow_html=True)
#     st.header("🛠 Control")
#     # Button to clear the FAISS database
#     if st.button("🗑 Clear DB"):
#         clear_db()
#         st.session_state.vs = None # Reset the vector store in session state
#         st.success("DB cleared")
#     # Button to clear the chat history
#     if st.button("🧹 Clear Chat"):
#         st.session_state.msgs = []
#         st.success("Chat cleared")

# # --- Main Application Logic ---

# # Initialize session state variables if they don't exist
# if "vs" not in st.session_state:
#     st.session_state.vs = load_existing_index() # Load existing index on app start
# if "msgs" not in st.session_state:
#     st.session_state.msgs = [] # Initialize chat messages list

# # Process uploaded PDFs if 'Extract & Index' button is clicked and files are uploaded
# if run and uploaded:
#     st.session_state.msgs = [] # Clear chat messages for new processing
#     with st.spinner("Processing documents and building index..."):
#         st.session_state.vs = load_and_index(uploaded) # Load and index the new PDFs
#     if st.session_state.vs:
#         # Add a confirmation message to the chat
#         st.session_state.msgs.append({
#             "role": "assistant",
#             "content": "Extraction & indexing done. Ask anything!"
#         })

# # Display previous chat messages
# for msg in st.session_state.msgs:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # Chat input for user queries
# if query := st.chat_input("Ask about the PDF content or tables..."):
#     # Add user query to chat history
#     st.session_state.msgs.append({"role": "user", "content": query})
#     with st.chat_message("user"):
#         st.markdown(query) # Display user query in chat UI

#     if st.session_state.vs: # Only proceed if vector store is available
#         chain = get_chat_chain(st.session_state.vs) # Get the RAG chain
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."): # Show spinner while LLM is processing
#                 # Stream the response from the LLM
#                 resp = "".join(chain.stream(query))
#                 st.markdown(resp) # Display LLM response
#                 st.session_state.msgs.append({"role": "assistant", "content": resp}) # Add response to chat history
#     else:
#         st.error("Please upload and process PDFs first to enable chat functionality.")







# import os, io, tempfile, shutil, csv, json, fitz, traceback, base64
# import streamlit as st
# from PIL import Image
# import pdfplumber
# from concurrent.futures import ThreadPoolExecutor
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_ollama import ChatOllama
# from langchain_core.messages import HumanMessage

# # SETTINGS
# OLLAMA_BASE_URL = "http://localhost:11434"
# QA_MODEL = "llama3:latest"
# CLEANER_MODEL = "llama3:latest"
# VISION_MODEL = "llama3.2-vision:latest"
# EMBED_MODEL = "nomic-embed-text"
# DB_DIR = "./faiss_db"

# st.set_page_config(page_title="PDF QA with LLaMA3", layout="wide")
# st.title("📄 PDF QA with Table Extraction & RAG")

# @st.cache_resource(show_spinner="Loading models...")
# def load_models():
#     qa_llm = ChatOllama(model=QA_MODEL, base_url=OLLAMA_BASE_URL)
#     cleaner_llm = ChatOllama(model=CLEANER_MODEL, base_url=OLLAMA_BASE_URL)
#     vision_llm = ChatOllama(model=VISION_MODEL, base_url=OLLAMA_BASE_URL)
#     return qa_llm, cleaner_llm, vision_llm

# qa_llm, cleaner_llm, vision_llm = load_models()

# def pdf_to_images(pdf_path):
#     doc = fitz.open(pdf_path)
#     return [Image.frombytes("RGB", [p.get_pixmap(dpi=150).width, p.get_pixmap(dpi=150).height], p.get_pixmap(dpi=150).samples) for p in doc]

# def extract_text_and_tables_pdfplumber(path):
#     texts, tables = [], []
#     with pdfplumber.open(path) as pdf:
#         for i, page in enumerate(pdf.pages):
#             text = page.extract_text()
#             if text:
#                 texts.append((i, text))
#             extracted_tables = page.extract_tables()
#             if extracted_tables:
#                 for t in extracted_tables:
#                     if t: tables.append(t)
#     return texts, tables

# def process_scanned_page(i, img, fname):
#     try:
#         img_buffer = io.BytesIO()
#         img.save(img_buffer, format="PNG")
#         img_bytes = img_buffer.getvalue()
#         img_b64 = base64.b64encode(img_bytes).decode()

#         message = HumanMessage(content=[
#             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
#             {"type": "text", "text": "Read the image and extract all text and tables (in CSV if present). Do not explain."}
#         ])
#         response = vision_llm.invoke([message])
#         return f"--- Page {i+1} ---\n{response.content.strip()}"
#     except Exception:
#         return f"Error processing page {i+1} of {fname}:\n{traceback.format_exc()}"

# def build_index(files, is_scanned):
#     all_docs = []
#     for f in files:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#             tmp.write(f.read())
#             tmp_path = tmp.name

#         st.write(f"Processing {f.name}...")
#         if is_scanned:
#             images = pdf_to_images(tmp_path)
#             with st.spinner("Running OCR on scanned pages..."):
#                 with ThreadPoolExecutor(max_workers=3) as executor:
#                     futures = [executor.submit(process_scanned_page, i, img, f.name) for i, img in enumerate(images)]
#                     raw_extracted = [future.result() for future in futures]

#             full_content = "\n".join(raw_extracted)
#             clean_prompt = f"""
# Below is messy OCR and partial table data extracted from a scanned PDF document.
# Some tables may be split across pages or contain merged cells.
# Your task is to clean and stitch related tables together, and output each complete table in valid CSV format.
# Only output the cleaned tables in CSV. Do not explain.
# ---
# {full_content}
# ---
# Cleaned Tables (CSV Only):
# """
#             try:
#                 cleaned_output = cleaner_llm.invoke(clean_prompt)
#                 all_docs.append(Document(page_content=cleaned_output, metadata={"source": f"{f.name}-cleaned"}))
#             except Exception:
#                 st.warning(f"Table cleaning failed for {f.name}:\n{traceback.format_exc()}")

#         else:
#             texts, tables = extract_text_and_tables_pdfplumber(tmp_path)
#             for idx, text in texts:
#                 all_docs.append(Document(page_content=text, metadata={"source": f"{f.name}-p{idx}"}))

#             raw_table_str = "\n---\n".join(["\n".join(["\t".join(map(str, row)) for row in t]) for t in tables])
#             stitch_prompt = f'''
# Below are raw tables extracted from a digital PDF.
# Some tables may be split across pages or contain merged cells.
# Your task is to clean and merge any related or partial tables.
# Return all final tables in valid CSV format only. Do not explain.
# ---
# {raw_table_str}
# ---
# Cleaned and Merged Tables (CSV Only):
# '''
#             try:
#                 stitched_csv = cleaner_llm.invoke(stitch_prompt)
#                 all_docs.append(Document(page_content=stitched_csv, metadata={"source": f"{f.name}-cleaned"}))
#             except Exception:
#                 st.warning(f"Failed to stitch tables for {f.name}:\n{traceback.format_exc()}")

#         os.remove(tmp_path)

#     if not all_docs:
#         st.warning("No content to index.")
#         return None

#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     docs = splitter.split_documents(all_docs)
#     embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
#     vs = FAISS.from_documents(docs, embeddings)

#     if os.path.exists(DB_DIR): shutil.rmtree(DB_DIR)
#     vs.save_local(DB_DIR)
#     return vs

# def get_rag_chain(vs):
#     prompt = ChatPromptTemplate.from_template(
#         "Answer the question based only on the context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
#     )
#     return {"context": vs.as_retriever(), "question": RunnablePassthrough()} | prompt | qa_llm | StrOutputParser()

# # --- SIDEBAR ---
# with st.sidebar:
#     st.header("📁 PDF Controls")
#     scanned = st.checkbox("Scanned PDFs (OCR)")
#     files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")

#     col1, col2 = st.columns(2)
#     if col1.button("🔄 Build Index") and files:
#         with st.spinner("Indexing..."):
#             st.session_state.vs = build_index(files, scanned)
#             if st.session_state.vs:
#                 st.session_state.chat = [{"role": "assistant", "content": "Index created. Ask your questions!"}]
#         st.success("Index built.")
#         st.rerun()

#     if col2.button("🗑 Clear DB"):
#         if os.path.exists(DB_DIR): shutil.rmtree(DB_DIR)
#         st.session_state.vs = None
#         st.success("Database cleared.")

#     if st.button("🧹 Clear Chat"):
#         st.session_state.chat = []

# # --- CHAT INTERFACE ---
# if "vs" not in st.session_state: st.session_state.vs = None
# if "chat" not in st.session_state: st.session_state.chat = []

# if st.session_state.vs is None and os.path.exists(DB_DIR):
#     embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
#     st.session_state.vs = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)

# for msg in st.session_state.chat:
#     with st.chat_message(msg["role"]): st.markdown(msg["content"])

# query = st.chat_input("Ask a question about your PDFs...")
# if query:
#     st.chat_message("user").markdown(query)
#     st.session_state.chat.append({"role": "user", "content": query})

#     if st.session_state.vs:
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 try:
#                     chain = get_rag_chain(st.session_state.vs)
#                     answer = chain.invoke({"question": query})
#                     st.markdown(answer)
#                     st.session_state.chat.append({"role": "assistant", "content": answer})
#                 except Exception:
#                     st.error("❌ Error answering your query:\n" + traceback.format_exc())
#     else:
#         st.info("⚠ Please build an index from a PDF before chatting.")