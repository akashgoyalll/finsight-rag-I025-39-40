import os, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("EMBEDDINGS", "tfidf")
from langchain_core.documents import Document
from finsight.loader import load_pdf
from finsight.indexing import chunk, build_vectorstore, get_retriever
from finsight.chain import parser, format_docs, resolve_sources, FinancialAnswer

PDF = ROOT / "data" / "nke-10k-2023.pdf"


@pytest.fixture(scope="module")
def pages():
    return load_pdf(PDF)


def test_loader_attaches_source_and_page(pages):
    assert len(pages) == 107
    assert pages[0].metadata["source"] == "nke-10k-2023.pdf"
    assert pages[0].metadata["page"] == 1


def test_metadata_survives_chunking(pages):
    chunks = chunk(pages)
    assert len(chunks) > len(pages)
    assert all({"source", "page", "start_index"} <= c.metadata.keys() for c in chunks)


def test_retriever_returns_cited_chunks():
    vs, _, _ = build_vectorstore([PDF])
    docs = get_retriever(vs, k=4).invoke("How many employees did NIKE have?")
    assert len(docs) == 4 and all("page" in d.metadata for d in docs)


def test_parser_rejects_malformed_output():
    with pytest.raises(Exception):
        parser.parse("Revenue was about 51 billion")  # free text, not the JSON schema


def test_sources_resolved_from_metadata_not_llm():
    docs = [Document(page_content="alpha", metadata={"source": "f.pdf", "page": 7}),
            Document(page_content="beta", metadata={"source": "f.pdf", "page": 9})]
    parsed = FinancialAnswer(answer="x", answerable=True, used_excerpts=[2, 5])  # 5 is out of range
    out = resolve_sources({"parsed": parsed, "docs": docs})
    assert [s["page"] for s in out["sources"]] == [9]  # invalid excerpt id dropped
    assert "[1] (f.pdf, p.7)" in format_docs(docs)


def test_full_lcel_chain_and_memory_wiring():
    """Plumbing test only: a scripted fake LLM checks retriever->prompt->LLM->parser->memory
    executes end to end. It is NOT evidence of answer quality."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from finsight.chain import build_chain, with_memory, _store
    vs, _, _ = build_vectorstore([PDF])
    fake = FakeListChatModel(responses=[
        '{"answer": "stub", "answerable": true, "used_excerpts": [1]}',
        '{"answer": "stub2", "answerable": false, "used_excerpts": []}'])
    chat = with_memory(build_chain(get_retriever(vs, k=4), fake))
    cfg = {"configurable": {"session_id": "t"}}
    out = chat.invoke({"question": "How many employees did NIKE have?"}, config=cfg)
    assert out["sources"][0]["source"] == "nke-10k-2023.pdf" and "page" in out["sources"][0]
    chat.invoke({"question": "And the year before?"}, config=cfg)
    assert len(_store["t"].messages) == 4  # 2 human + 2 AI turns retained
