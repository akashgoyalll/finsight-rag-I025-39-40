"""Full RAG question answering (needs an LLM).

  export LLM_MODEL="<provider>:<model>"   # init_chat_model format, e.g. "openai:<model-name>"
  export <PROVIDER>_API_KEY=...           # key for that provider
  python scripts/ask.py "What were NIKE's total revenues in fiscal 2023?"
"""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from langchain.chat_models import init_chat_model
from finsight.indexing import build_vectorstore, get_retriever
from finsight.chain import build_chain, with_memory

if "LLM_MODEL" not in os.environ:
    sys.exit("Set LLM_MODEL (e.g. 'openai:<model>') and the provider's API key first.")
vs, _, _ = build_vectorstore([ROOT / "data" / "nke-10k-2023.pdf"])
llm = init_chat_model(os.environ["LLM_MODEL"], temperature=0)
chat = with_memory(build_chain(get_retriever(vs), llm))
for q in sys.argv[1:] or ["What were NIKE, Inc.'s total revenues in fiscal 2023?"]:
    out = chat.invoke({"question": q}, config={"configurable": {"session_id": "cli"}})
    print(json.dumps(out, indent=2))
