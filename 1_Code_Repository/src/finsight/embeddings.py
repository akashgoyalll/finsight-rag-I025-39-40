"""Embedding model factory.

- "hf"    : sentence-transformers/all-MiniLM-L6-v2 via langchain-huggingface (default; used in Colab)
- "tfidf" : lexical (keyword) embeddings from scikit-learn TF-IDF. Used (a) as the keyword half of the
            Phase 3 hybrid retriever and (b) as an offline fallback when models cannot be downloaded.
            It is NOT a semantic model.
"""
import os
from langchain_core.embeddings import Embeddings


class TfidfEmbeddings(Embeddings):
    def __init__(self, corpus: list[str], max_features: int = 4096):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                                   stop_words="english", sublinear_tf=True)
        self.vec.fit(corpus)

    def _embed(self, texts):
        from sklearn.preprocessing import normalize
        return normalize(self.vec.transform(texts)).toarray().tolist()

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]


def get_embeddings(corpus: list[str] | None = None, kind: str | None = None) -> Embeddings:
    """`kind` overrides the EMBEDDINGS environment variable ("hf" | "tfidf")."""
    kind = (kind or os.getenv("EMBEDDINGS", "hf")).lower()
    if kind == "tfidf":
        if corpus is None:
            raise ValueError("TF-IDF embeddings must be fitted on the chunk corpus")
        return TfidfEmbeddings(corpus)
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
