"""Ask FinSight-RAG questions from the terminal (Phase 3 system: hybrid retrieval + Gemini).

  export GOOGLE_API_KEY=...                         # never commit this
  export LLM_MODEL="google_genai:gemini-3.5-flash"  # optional (default); any init_chat_model string works
  python scripts/ask.py "What were NIKE's total revenues in fiscal 2023?" "How much did that grow on a currency-neutral basis?"

All questions on one command line share one conversation (memory), so follow-ups work.
"""
import json, sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from langchain_core._api import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
from finsight.indexing import load_pages, chunk, index_chunks
from finsight.hybrid import build_hybrid_retriever
from finsight.chain import build_chain, with_memory
from finsight.llm import get_llm

chunks = chunk(load_pages([ROOT / "data" / "nke-10k-2023.pdf"]))
retriever = build_hybrid_retriever(index_chunks(chunks), index_chunks(chunks, kind="tfidf"))
chat = with_memory(build_chain(retriever, get_llm()))
for q in sys.argv[1:] or ["What were NIKE, Inc.'s total revenues in fiscal 2023?"]:
    out = chat.invoke({"question": q}, config={"configurable": {"session_id": "cli"}})
    print(f"\nQ: {q}")
    print(json.dumps(out, indent=2, ensure_ascii=False))
