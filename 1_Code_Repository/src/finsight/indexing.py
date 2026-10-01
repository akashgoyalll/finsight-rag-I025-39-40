"""Chunking + vector store construction (metadata is preserved through splitting).

Phase 3 fix: every in-memory index now gets its own Chroma collection name. Chroma's in-process
clients share state, so building the index twice in one Python process (e.g. re-running a notebook
cell, or building a semantic and a keyword index) used to append to the same "filings" collection:
516 chunks became 1032 and the retriever returned duplicate chunks.
"""
import os
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from .loader import load_pdf
from .embeddings import get_embeddings

CHUNK_SIZE, CHUNK_OVERLAP = 1000, 200


def load_pages(pdf_paths):
    return [d for p in pdf_paths for d in load_pdf(p)]


def chunk(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True)
    return splitter.split_documents(docs)  # split_documents() keeps source/page metadata


def index_chunks(chunks, kind: str | None = None, persist_dir=None, collection_name=None):
    """Embed `chunks` with the given backend ("hf" | "tfidf") into a fresh Chroma collection."""
    kind = (kind or os.getenv("EMBEDDINGS", "hf")).lower()
    emb = get_embeddings(corpus=[c.page_content for c in chunks], kind=kind)
    name = collection_name or f"filings-{kind}-{uuid.uuid4().hex[:8]}"
    return Chroma.from_documents(chunks, emb, collection_name=name, persist_directory=persist_dir)


def build_vectorstore(pdf_paths, persist_dir=None, kind: str | None = None):
    pages = load_pages(pdf_paths)
    chunks = chunk(pages)
    vs = index_chunks(chunks, kind=kind, persist_dir=persist_dir)
    return vs, pages, chunks


def get_retriever(vs, k: int = 4):
    return vs.as_retriever(search_kwargs={"k": k})
