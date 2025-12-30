import fitz  # PyMuPDF

def extract_text(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages.append({
                "text": text,
                "page": page_number
            })

    return pages




