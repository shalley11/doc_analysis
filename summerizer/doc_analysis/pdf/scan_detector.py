import pdfplumber
from doc_analysis.config import MIN_TEXT_CHARS

def is_scanned_pdf(pdf_path: str, max_pages: int = 3) -> bool:
    total = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:max_pages]:
            total += len(page.extract_text() or "")
    return total < MIN_TEXT_CHARS
