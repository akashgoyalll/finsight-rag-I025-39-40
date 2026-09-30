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
        'How many employees did NIKE have the year before?',          # condensed follow-up
        '{"answer": "stub2", "answerable": false, "used_excerpts": []}'])
    chat = with_memory(build_chain(get_retriever(vs, k=4), fake))
    cfg = {"configurable": {"session_id": "t"}}
    out = chat.invoke({"question": "How many employees did NIKE have?"}, config=cfg)
    assert out["sources"][0]["source"] == "nke-10k-2023.pdf" and "page" in out["sources"][0]
    out2 = chat.invoke({"question": "And the year before?"}, config=cfg)
    assert out2["retrieval_query"] == "How many employees did NIKE have the year before?"
    assert len(_store["t"].messages) == 4  # 2 human + 2 AI turns retained


# ----------------------------------------------------------------------------- Phase 3 tests
class _RecordingRetriever:
    """Minimal stand-in retriever that records the queries it receives."""
    def __new__(cls, docs):
        from langchain_core.retrievers import BaseRetriever

        class R(BaseRetriever):
            seen: list = []
            def _get_relevant_documents(self, query, *, run_manager):
                self.seen.append(query)
                return docs
        return R(seen=[])


def _doc(page, start, text="x"):
    return Document(page_content=f"{text}-{page}-{start}",
                    metadata={"source": "f.pdf", "page": page, "start_index": start})


def test_rrf_fuses_ranked_lists_and_dedupes():
    from finsight.hybrid import reciprocal_rank_fusion
    a, b, c, d = _doc(1, 0), _doc(2, 0), _doc(3, 0), _doc(4, 0)
    fused = reciprocal_rank_fusion([[a, b, c], [c, d, a]], k=4)
    pages = [x.metadata["page"] for x in fused]
    assert len(pages) == len(set(pages)) == 4        # a and c counted once
    assert set(pages[:2]) == {1, 3}                  # found by both retrievers -> ranked first
    assert pages[-1] in (2, 4)


def test_hybrid_retriever_is_a_langchain_retriever():
    from finsight.hybrid import HybridRetriever
    r = HybridRetriever(retrievers=[_RecordingRetriever([_doc(1, 0)]),
                                    _RecordingRetriever([_doc(2, 0), _doc(1, 0)])], k=6)
    out = r.invoke("q")
    assert [x.metadata["page"] for x in out] == [1, 2]


def test_rebuilding_index_does_not_duplicate_chunks(pages):
    from finsight.indexing import index_chunks
    chunks = chunk(pages)
    v1 = index_chunks(chunks, kind="tfidf")
    v2 = index_chunks(chunks, kind="tfidf")
    assert v1._collection.count() == v2._collection.count() == len(chunks)  # was 2x before the fix


def test_followup_question_is_rewritten_before_retrieval():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from finsight.chain import build_chain, with_memory
    rec = _RecordingRetriever([_doc(36, 0, "Revenues rose 16% on a currency-neutral basis")])
    fake = FakeListChatModel(responses=[
        '{"answer": "$51.2 billion", "answerable": true, "used_excerpts": [1]}',
        "How much did NIKE's fiscal 2023 revenues grow on a currency-neutral basis?",
        '{"answer": "16%", "answerable": true, "used_excerpts": [1]}'])
    chat = with_memory(build_chain(rec, fake))
    cfg = {"configurable": {"session_id": "follow"}}
    chat.invoke({"question": "What were NIKE's total revenues in fiscal 2023?"}, config=cfg)
    chat.invoke({"question": "How much did that grow on a currency-neutral basis?"}, config=cfg)
    assert rec.seen == ["What were NIKE's total revenues in fiscal 2023?",
                        "How much did NIKE's fiscal 2023 revenues grow on a currency-neutral basis?"]


def test_llm_factory_sampling_params(monkeypatch):
    import finsight.llm as L
    calls = []
    monkeypatch.setattr(L, "init_chat_model", lambda m, **kw: calls.append((m, kw)))
    L.get_llm("google_genai:gemini-3.5-flash")
    L.get_llm("groq:some-model")
    assert calls == [("google_genai:gemini-3.5-flash", {}), ("groq:some-model", {"temperature": 0})]
