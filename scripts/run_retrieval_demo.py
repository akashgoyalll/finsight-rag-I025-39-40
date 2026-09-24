"""Phase 2 evidence: build the index and run the test questions through the retriever.

Usage:  EMBEDDINGS=hf    python scripts/run_retrieval_demo.py   (Colab / default)
        EMBEDDINGS=tfidf python scripts/run_retrieval_demo.py   (offline fallback)
No LLM is called here; this validates load -> chunk -> embed -> store -> retrieve -> prompt.
"""
import json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finsight.indexing import build_vectorstore, get_retriever, CHUNK_SIZE, CHUNK_OVERLAP
from finsight.chain import prompt, format_docs

PDF = ROOT / "data" / "nke-10k-2023.pdf"
K = 4

t0 = time.time()
vs, pages, chunks = build_vectorstore([PDF])
retriever = get_retriever(vs, k=K)
print(f"Embeddings backend : {os.getenv('EMBEDDINGS', 'hf')}")
print(f"Pages loaded       : {len(pages)}  ({PDF.name})")
print(f"Chunks created     : {len(chunks)}  (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
print(f"Sample chunk meta  : {chunks[100].metadata}")
print(f"Index build time   : {time.time()-t0:.1f}s   | retriever top-k = {K}\n")

qs = json.loads((ROOT / "tests" / "questions.json").read_text())
results = []
for q in qs:
    docs = retriever.invoke(q["question"])
    got_pages = [d.metadata["page"] for d in docs]
    text = " ".join(d.page_content for d in docs)
    if "evidence_all" in q:   # needs every piece (e.g. numerator AND denominator)
        found = [e for e in q["evidence_all"] if e in text]
        hit = len(found) == len(q["evidence_all"])
        note = f"needs {q['evidence_all']}; found {found}"
    else:
        found = [e for e in q["evidence_any"] if e.lower() in text.lower()]
        hit = bool(found)
        note = f"evidence {'found: ' + found[0] if hit else 'NOT found'}"
    results.append({**q, "retrieved_pages": got_pages, "evidence_retrieved": hit, "note": note})
    print(f"{q['id']} [{q['type']}]\n   Q: {q['question']}\n"
          f"   truth : {q['ground_truth']}\n"
          f"   retrieved pages: {got_pages}   -> {'HIT ' if hit else 'MISS'} ({note})")

n_hit = sum(r["evidence_retrieved"] for r in results)
print(f"\nRetrieval evidence hit-rate: {n_hit}/{len(results)}")

# Show the exact grounded prompt that would be sent to the LLM for Q1
docs = retriever.invoke(qs[0]["question"])
msgs = prompt.format_messages(context=format_docs(docs), question=qs[0]["question"], history=[])
sys_txt = msgs[0].content
print("\n--- Grounded prompt preview (Q1, system message, truncated) ---")
print(sys_txt[:900] + "\n  [...]")

out = ROOT / "outputs"
out.mkdir(exist_ok=True)
(out / f"phase2_retrieval_results_{os.getenv('EMBEDDINGS','hf')}.json").write_text(json.dumps(results, indent=2))
