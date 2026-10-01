"""PDF loader for financial filings.

Replaces langchain_community.document_loaders.PyPDFLoader (langchain-community
was sunset in 2026). Produces one LangChain Document per PDF page with the same
metadata used for citation: `source` (file name) and `page` (1-based page
number, as shown in a PDF viewer).
"""
from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document


def load_pdf(path: str | Path) -> list[Document]:
    path = Path(path)
    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue  # skip blank/scanned pages (pypdf has no OCR)
        docs.append(Document(
            page_content=text,
            metadata={"source": path.name, "page": i + 1, "total_pages": len(reader.pages)},
        ))
    return docs
