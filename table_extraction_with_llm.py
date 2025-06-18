import fitz  # PyMuPDF
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def extract_tables_with_llm(pdf_path, model="mistral:7b", base_url="http://localhost:11434", max_chars=12000):
    """
    Extracts tables from PDF using an LLM via LangChain.
    Args:
        pdf_path (str): Path to the PDF file.
        model (str): Name of the Ollama model (default: mistral:7b).
        base_url (str): URL for the local Ollama instance.
        max_chars (int): Max characters to send to the LLM to avoid overload.

    Returns:
        str: Extracted tables as CSV-style text.
    """
    try:
        doc = fitz.open(pdf_path)
        all_text = "\n".join([page.get_text() for page in doc])
        all_text = all_text.strip()

        if not all_text:
            return "No extractable text found in PDF."

        # Truncate to avoid LLM context limit overflow
        truncated_text = all_text[:max_chars]

        prompt = ChatPromptTemplate.from_template(
            """You are an expert at extracting tabular data from messy PDF content.

Below is some raw text extracted from a PDF file. It may be noisy or include irrelevant data.

Your task:
- Extract all meaningful tabular data only.
- Format each table as plain CSV-style text (comma-separated, no index).
- Ignore non-tabular content and avoid repeating context.
- Do not invent data.

Raw PDF Text:
{context}

Tables (CSV format):
"""
        )

        chain = (
            prompt
            | ChatOllama(model=model, base_url=base_url, temperature=0.2)
            | StrOutputParser()
        )

        result = chain.invoke({"context": truncated_text})
        return result.strip() if result else "No tables found."
    
    except Exception as e:
        return f"LLM extraction failed: {e}"
