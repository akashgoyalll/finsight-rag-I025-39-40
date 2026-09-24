# FinSight-RAG
### A Source-Grounded Question-Answering Assistant for Corporate Financial Filings

**Lab 9 PSIS — Activity 1: RAG-Based Domain Assistant (Financial Report Analyzer)**
SVKM's NMIMS MPSTME, B.Tech AI · Group of 3 · **Status: Phase 2 — partially working system**

FinSight-RAG answers natural-language questions about a company's annual report / SEC 10-K and
returns every answer with the **document name and page number** of the supporting passage.

## Architecture
```
PDF filing ─► load_pdf (pypdf; metadata: source, page)
          ─► RecursiveCharacterTextSplitter (1000 chars / 200 overlap; metadata preserved)
          ─► Embeddings (all-MiniLM-L6-v2)  ─► Chroma vector store
question ─► VectorStoreRetriever (top-k=4) ─► ChatPromptTemplate (numbered, cited excerpts + history)
          ─► LLM ─► PydanticOutputParser {answer, answerable, used_excerpts}
          ─► source resolution from chunk metadata ─► {answer, answerable, sources[doc, page, passage]}
          (wrapped in RunnableWithMessageHistory for multi-turn memory)
```

| LangChain component | Where | Status |
|---|---|---|
| VectorStoreRetriever (Chroma) | `src/finsight/indexing.py` | ✅ working |
| ChatPromptTemplate | `src/finsight/chain.py` | ✅ working (prompt verified) |
| LCEL composition (`\|`, `RunnablePassthrough.assign`) | `src/finsight/chain.py` | ✅ wired & tested |
| PydanticOutputParser | `src/finsight/chain.py` | ✅ tested |
| RunnableWithMessageHistory (memory) | `src/finsight/chain.py` | ✅ wired & tested |

Citations are attached **from chunk metadata**, not written by the LLM — the model only reports which
numbered excerpts it used, so it cannot fabricate a page number.

## Quick start (terminal)
```bash
git clone https://github.com/akashgoyalll/finsight-rag-I025-39-40.git && cd finsight-rag
pip install -r requirements.txt
bash scripts/get_data.sh
python scripts/run_retrieval_demo.py                  # retrieval evaluation (no LLM needed)
EMBEDDINGS=tfidf python scripts/run_retrieval_demo.py # offline fallback, no model download
python -m pytest tests/ -v                            # 6 unit/integration tests
# Phase 3 — full answers (needs an LLM key):
pip install langchain-<provider> && export LLM_MODEL="<provider>:<model>" <PROVIDER>_API_KEY=...
python scripts/ask.py "What were NIKE's total revenues in fiscal 2023?"
```
Colab: [open the notebook](https://colab.research.google.com/github/akashgoyalll/finsight-rag-I025-39-40/blob/main/notebooks/FinSight_RAG_Phase2.ipynb) · executed run with outputs: `notebooks/FinSight_RAG_Phase2_executed.ipynb`.

## Data
Development filing: **NIKE, Inc. FY2023 Form 10-K** (107 pages) — the example filing from LangChain's official
RAG tutorial. Any text-based annual report / 10-K PDF can be dropped into `data/`. Scanned PDFs are not supported
(pypdf has no OCR).

## Test questions and Phase 2 retrieval results
`tests/questions.json` has 6 questions whose answers were checked against the filing, including one deliberate
weak case (Q6: a segment EBIT-margin calculation that needs numbers from two tables and is not stated in the filing).
No LLM is called at this stage. **HIT** = the top-4 retrieved chunks contain the evidence string needed to answer.

| Q | Question type | Semantic (all-MiniLM-L6-v2, Colab) | Lexical (TF-IDF) |
|---|---|---|---|
| Q1 | total revenues | HIT (pp. 36, 89, 31, 32) | MISS |
| Q2 | employee count | MISS (p. 9 not retrieved) | HIT |
| Q3 | gross-margin change | HIT | HIT |
| Q4 | NIKE Direct growth | HIT | HIT |
| Q5 | FX risk (narrative) | MISS* | HIT |
| Q6 | segment EBIT margin (weak case) | HIT (both numbers from p. 39) | MISS |
| | **Automatic hit-rate** | **4/6** | **4/6** |

\* False negative of the string check: p. 46 (retrieved) discusses NIKE's transactional foreign-currency
exposures and hedging, which is relevant, but it does not contain the exact phrase being matched.

The two methods fail on different questions, and between them they retrieve the evidence for all 6. That is the evidence
behind the improvement we plan to test in Phase 3: hybrid (semantic + keyword) retrieval. Logs: `outputs/`.

## Repository layout
```
src/finsight/  loader.py · embeddings.py · indexing.py · chain.py
scripts/       run_retrieval_demo.py · ask.py · get_data.sh
tests/         test_pipeline.py · questions.json
notebooks/     FinSight_RAG_Phase2.ipynb · FinSight_RAG_Phase2_executed.ipynb (Colab run)
outputs/       run logs and JSON results
```
