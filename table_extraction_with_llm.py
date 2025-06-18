import fitz
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def extract_tables_with_llm(pdf_path, model="mistral:7b", base_url="http://localhost:11434", max_chars=12000):
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc).strip()
    if not text: return "No text found in PDF."

    prompt = ChatPromptTemplate.from_template(
        """Extract all tables from the following PDF text. Provide results in CSV format, one line per row, no index.\n\n{text}\n\nCSV:"""
    )
    chain = prompt | ChatOllama(model=model, base_url=base_url) | StrOutputParser()
    return chain.invoke({"text": text[:max_chars]}).strip()
