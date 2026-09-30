"""Hybrid (semantic + keyword) retrieval with Reciprocal Rank Fusion (RRF).

Why: in Phase 2 the semantic retriever (all-MiniLM-L6-v2) and the keyword retriever (TF-IDF) failed on
different questions (semantic missed the exact employee count; keyword was misled by segment tables),
and the union of their top-4 results contained the evidence for all six test questions.

How: each retriever returns its own ranked list; a chunk's fused score is
    sum over lists of  weight / (rrf_k + rank)
(Cormack, Clarke & Buettcher, SIGIR 2009; rrf_k = 60 as in the paper). Scores from the two
retrievers are not on the same scale, so only ranks are used. Chunks are identified by
(source, page, start_index), so the same chunk found by both retrievers is counted once.

Written as a LangChain BaseRetriever, so it plugs into the LCEL chain exactly like the
VectorStoreRetriever it replaces. (LangChain's own EnsembleRetriever/BM25Retriever are not used:
BM25Retriever lived in the sunset langchain-community package.)
"""
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


def chunk_key(d: Document) -> tuple:
    m = d.metadata
    return (m.get("source"), m.get("page"), m.get("start_index"), d.page_content[:50])


def reciprocal_rank_fusion(ranked_lists: list[list[Document]], k: int,
                           rrf_k: int = 60, weights: list[float] | None = None) -> list[Document]:
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[tuple, float] = {}
    docs: dict[tuple, Document] = {}
    for w, ranked in zip(weights, ranked_lists):
        for rank, d in enumerate(ranked, 1):
            key = chunk_key(d)
            scores[key] = scores.get(key, 0.0) + w / (rrf_k + rank)
            docs.setdefault(key, d)
    order = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [docs[key] for key in order[:k]]


class HybridRetriever(BaseRetriever):
    """Fuses the ranked outputs of several retrievers (here: semantic + keyword) with RRF."""
    retrievers: list[BaseRetriever]
    weights: list[float] | None = None
    k: int = 6
    rrf_k: int = 60

    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        ranked = [r.invoke(query, config={"callbacks": run_manager.get_child()})
                  for r in self.retrievers]
        return reciprocal_rank_fusion(ranked, k=self.k, rrf_k=self.rrf_k, weights=self.weights)


def build_hybrid_retriever(semantic_vs, keyword_vs, fetch_k: int = 4, k: int = 6) -> HybridRetriever:
    """Top-`fetch_k` from each index, fused, best `k` returned to the prompt."""
    return HybridRetriever(
        retrievers=[semantic_vs.as_retriever(search_kwargs={"k": fetch_k}),
                    keyword_vs.as_retriever(search_kwargs={"k": fetch_k})],
        k=k)
