"""Embedding model factory.

- "hf"    : sentence-transformers/all-MiniLM-L6-v2 via langchain-huggingface (default; used in Colab)
- "tfidf" : offline lexical fallback (scikit-learn) for environments that cannot download models.
            NOT a semantic model; used only so the pipeline can be exercised offline.
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


def get_embeddings(corpus: list[str] | None = None) -> Embeddings:
    kind = os.getenv("EMBEDDINGS", "hf").lower()
    if kind == "tfidf":
        if corpus is None:
            raise ValueError("TF-IDF fallback must be fitted on the chunk corpus")
        return TfidfEmbeddings(corpus)
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
