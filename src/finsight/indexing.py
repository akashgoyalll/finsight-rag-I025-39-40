"""Chunking + vector store construction (metadata is preserved through splitting)."""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from .loader import load_pdf
from .embeddings import get_embeddings

CHUNK_SIZE, CHUNK_OVERLAP = 1000, 200


def chunk(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True)
    return splitter.split_documents(docs)  # split_documents() keeps source/page metadata


def build_vectorstore(pdf_paths, persist_dir=None):
    pages = [d for p in pdf_paths for d in load_pdf(p)]
    chunks = chunk(pages)
    emb = get_embeddings(corpus=[c.page_content for c in chunks])
    vs = Chroma.from_documents(chunks, emb, collection_name="filings",
                               persist_directory=persist_dir)
    return vs, pages, chunks


def get_retriever(vs, k: int = 4):
    return vs.as_retriever(search_kwargs={"k": k})
