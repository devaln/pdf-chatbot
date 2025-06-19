import fitz
import numpy as np
from PIL import Image
import easyocr

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize EasyOCR reader. It's important to initialize this once.
ocr_reader = easyocr.Reader(['en'], gpu=False)

def extract_text_mixed(pdf_path: str) -> str:
    """
    Extracts text from a PDF, prioritizing rule-based extraction and falling back to OCR
    for pages where rule-based extraction yields no text.
    Args:
        pdf_path (str): The path to the PDF file.
    Returns:
        str: A concatenated string of all extracted text, noting the source (text or OCR) per page.
    """
    doc = fitz.open(pdf_path) # Open the PDF document using PyMuPDF
    pages = []
    for i, page in enumerate(doc):
        # Attempt rule-based text extraction first
        txt = page.get_text().strip()
        if txt:
            pages.append(f"Page {i+1} Text:\n{txt}") # If text found, append it
        else:
            # If no text, fall back to OCR
            try:
                pix = page.get_pixmap() # Get a pixel map (image) of the page
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples) # Convert to PIL Image
                ocr = ocr_reader.readtext(np.array(img), detail=0) # Perform OCR
                tmp = "\n".join([l.strip() for l in ocr if l.strip()]) # Join non-empty OCR results
                pages.append(f"Page {i+1} OCR:\n{tmp}") # Append OCR text
            except Exception as e:
                pages.append(f"Page {i+1} OCR failed: {e}") # Log OCR failure
    return "\n\n".join(pages)


def extract_tables_with_llm(pdf_path: str, model: str, base_url: str, max_chars: int = 12000) -> str:
    """
    Extracts structured tables from PDF content using an LLM.
    The PDF content is first extracted (mixed text and OCR) and then
    fed to the LLM in chunks for table extraction.
    Args:
        pdf_path (str): The path to the PDF file.
        model (str): The name of the LLM model to use (e.g., "llama3:latest").
        base_url (str): The base URL for the Ollama server.
        max_chars (int): Maximum number of characters to send to the LLM in one chunk.
    Returns:
        str: A concatenated string of extracted tables in CSV format from the LLM.
             Returns error messages if content is not extractable or LLM fails.
    """
    try:
        combined = extract_text_mixed(pdf_path) # Get combined text from PDF
        if not combined.strip():
            return "No extractable content found in the PDF."

        # Split the combined text into smaller parts if it's too long for the LLM context window
        parts = [combined[i:i+max_chars] for i in range(0, len(combined), max_chars)]
        results = []
        for part in parts:
            # Define the prompt for table extraction
            prompt = ChatPromptTemplate.from_template(
                "You are a table extraction specialist.\n\nRaw Content:\n{context}\n\n"
                "Extract all tables present, return clean CSV, no commentary.\n\nCSV:"
            )
            # Create a LangChain chain: prompt -> LLM -> string parser
            chain = prompt | ChatOllama(model=model, base_url=base_url, temperature=0.2) | StrOutputParser()
            results.append(chain.invoke({"context": part}).strip()) # Invoke the chain and get the structured CSV
        return "\n\n".join(results) # Join results from all parts
    except Exception as e:
        return f"LLM extraction failed: {e}" # Return error if LLM extraction fails