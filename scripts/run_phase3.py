"""Phase 3 evaluation: retrieval comparison + LLM answers + grounding checks + multi-turn test.

  export GOOGLE_API_KEY=...                       # Gemini key (never commit it)
  export LLM_MODEL="google_genai:gemini-3.5-flash"  # optional; this is the default
  python scripts/run_phase3.py

Writes outputs/phase3_results.json and outputs/phase3_summary.md (dry run: *_dryrun.*).
--dry-run exercises the whole pipeline offline (TF-IDF for both halves, scripted fake LLM);
its "answers" are placeholders and are NOT results.

Automatic checks are deliberately simple and are backed by a manual grounding review:
  answer_ok      expected figure/phrase appears in the answer (answer_any / answer_all);
                 for Q7, the model must set answerable=false
  cited_ok       at least one cited chunk is on a ground-truth page, or its full text contains
                 the evidence string  (retrieval-grounding proxy)
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finsight.indexing import load_pages, chunk, index_chunks, get_retriever
from finsight.hybrid import build_hybrid_retriever, chunk_key
from finsight.chain import build_chain, with_memory, _store
from langchain_core._api import LangChainDeprecationWarning
# RunnableWithMessageHistory / InMemoryChatMessageHistory are deprecated in langchain-core 1.6
# (LangChain recommends LangGraph persistence). They still work; noted in the report as a limitation.
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--delay", type=float, default=4.0, help="seconds between LLM calls (free-tier rate limits)")
args = ap.parse_args()

PDF = ROOT / "data" / "nke-10k-2023.pdf"
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
qs = json.loads((ROOT / "tests" / "questions.json").read_text())
SEM = "tfidf" if args.dry_run else os.getenv("EMBEDDINGS", "hf")

# ---------------------------------------------------------------- 1. index (one chunk set, two indexes)
t0 = time.time()
pages = load_pages([PDF]); chunks = chunk(pages)
by_key = {chunk_key(c): c for c in chunks}
sem_vs = index_chunks(chunks, kind=SEM)
kw_vs = index_chunks(chunks, kind="tfidf")
print(f"Pages {len(pages)} | chunks {len(chunks)} | semantic index = {SEM} | keyword index = tfidf "
      f"| built in {time.time()-t0:.1f}s")
assert sem_vs._collection.count() == len(chunks) == kw_vs._collection.count(), "duplicate chunks in index"

retrievers = {
    "semantic@4": get_retriever(sem_vs, 4),
    "keyword@4": get_retriever(kw_vs, 4),
    "semantic@6": get_retriever(sem_vs, 6),
    "keyword@6": get_retriever(kw_vs, 6),
    "hybrid@4": build_hybrid_retriever(sem_vs, kw_vs, fetch_k=4, k=4),
    "hybrid@6": build_hybrid_retriever(sem_vs, kw_vs, fetch_k=4, k=6),
}


def evidence_hit(q, docs):
    text = " ".join(d.page_content for d in docs)
    if q.get("evidence_all"):
        return all(e in text for e in q["evidence_all"])
    return any(e.lower() in text.lower() for e in q.get("evidence_any", []))


# ---------------------------------------------------------------- 2. retrieval comparison (no LLM)
print("\n=== Retrieval comparison (Q1-Q6). string = evidence string in retrieved text; "
      "page = a ground-truth page retrieved ===")
retrieval = {}
answerable_qs = [q for q in qs if not q.get("expect_unanswerable")]
for name, r in retrievers.items():
    rows = []
    for q in answerable_qs:
        docs = r.invoke(q["question"])
        pg = [d.metadata["page"] for d in docs]
        rows.append({"id": q["id"], "pages": pg, "string_hit": evidence_hit(q, docs),
                     "page_hit": any(p in q["source_pages"] for p in pg)})
    retrieval[name] = rows
    s = sum(x["string_hit"] for x in rows); p = sum(x["page_hit"] for x in rows)
    print(f"{name:11s} string {s}/{len(rows)}  page {p}/{len(rows)}  | "
          + "  ".join(f"{x['id']}:{'H' if x['string_hit'] else 'm'}" for x in rows))

# ---------------------------------------------------------------- 3. LLM
if args.dry_run:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    llm = FakeListChatModel(responses=['{"answer": "DRY-RUN PLACEHOLDER", "answerable": true, "used_excerpts": [1]}'])
    model = "dry-run (FakeListChatModel)"; args.delay = 0
else:
    from finsight.llm import get_llm, model_name
    llm = get_llm(); model = model_name()
print(f"\nLLM: {model}")


def check(q, out):
    a = out.get("answer", "")
    if q.get("expect_unanswerable"):
        answer_ok = out.get("answerable") is False
    elif q.get("answer_check") == "manual":
        answer_ok = None
    elif q.get("answer_all"):
        answer_ok = all(s.lower() in a.lower() for s in q["answer_all"])
    else:
        answer_ok = any(s.lower() in a.lower() for s in q.get("answer_any", []))
    cited = out.get("sources", [])
    full = [by_key.get((s["source"], s["page"], s["start_index"], s["passage"][:50])) for s in cited]
    if q.get("expect_unanswerable"):
        cited_ok = None
    else:
        cited_ok = bool(cited) and (any(s["page"] in q["source_pages"] for s in cited)
                                    or evidence_hit(q, [c for c in full if c is not None]))
    return answer_ok, cited_ok


def ask(chat, question, session):
    t = time.time()
    try:
        out = chat.invoke({"question": question}, config={"configurable": {"session_id": session}})
    except Exception as e:  # parser failure, rate limit after retries, etc. -- recorded, not hidden
        out = {"answer": "", "answerable": None, "sources": [], "error": f"{type(e).__name__}: {e}"[:500]}
    out["latency_s"] = round(time.time() - t, 1)
    time.sleep(args.delay)
    return out


SYSTEMS = {"baseline (Phase 2: semantic@4)": "semantic@4", "final (Phase 3: hybrid@6)": "hybrid@6"}
answers = {}
for label, rname in SYSTEMS.items():
    chat = with_memory(build_chain(retrievers[rname], llm))
    print(f"\n=== Answers: {label} ===")
    rows = []
    for q in qs:
        out = ask(chat, q["question"], f"{rname}-{q['id']}")  # fresh session per question
        ok, cok = check(q, out)
        rows.append({"id": q["id"], "question": q["question"], "ground_truth": q["ground_truth"],
                     **out, "answer_ok": ok, "cited_ok": cok})
        print(f"\n{q['id']} {q['question']}\n  answer    : {out['answer']}\n"
              f"  answerable: {out.get('answerable')} | cited pages: {[s['page'] for s in out['sources']]}"
              f" | retrieved pages: {out.get('retrieved_pages')}\n"
              f"  auto-check: answer_ok={ok} cited_ok={cok} | {out['latency_s']}s"
              + (f"\n  ERROR: {out['error']}" if out.get("error") else ""))
    answers[label] = rows

# ---------------------------------------------------------------- 4. multi-turn (memory) test
FOLLOW = [("What were NIKE, Inc.'s total revenues in fiscal 2023?", None),
          ("How much did that grow on a currency-neutral basis?", ["16%", "16 percent"])]
multiturn = {}
for label, condense in [("without question rewriting (Phase 2)", False), ("with question rewriting (Phase 3)", True)]:
    chat = with_memory(build_chain(retrievers["hybrid@6"], llm, condense=condense))
    sid = f"mt-{condense}"; _store.pop(sid, None)
    turns = []
    for q, expect in FOLLOW:
        out = ask(chat, q, sid)
        out["answer_ok"] = None if expect is None else any(e in out.get("answer", "") for e in expect)
        turns.append({"question": q, **out})
    multiturn[label] = {"turns": turns, "history_messages": len(_store[sid].messages)}
    t2 = turns[1]
    print(f"\n=== Multi-turn, {label} ===\n  turn-2 retrieval query: {t2.get('retrieval_query')}\n"
          f"  turn-2 answer: {t2['answer']}\n  answer_ok (16%): {t2['answer_ok']} | "
          f"history messages: {multiturn[label]['history_messages']}")

# ---------------------------------------------------------------- 5. save
res = {"model": model, "dry_run": args.dry_run, "semantic_backend": SEM, "chunks": len(chunks),
       "pages": len(pages), "retrieval": retrieval, "answers": answers, "multiturn": multiturn,
       "generated": time.strftime("%Y-%m-%d %H:%M:%S %z")}
SUFFIX = "_dryrun" if args.dry_run else ""
(OUT / f"phase3_results{SUFFIX}.json").write_text(json.dumps(res, indent=2))

md = [f"# Phase 3 results\n\nModel: `{model}` · semantic backend: `{SEM}` · {len(pages)} pages, "
      f"{len(chunks)} chunks · generated {res['generated']}\n"]
if args.dry_run:
    md.append("> **DRY RUN — placeholder answers from a scripted fake model. Not results.**\n")
md.append("## Retrieval (Q1–Q6)\n\n| Retriever | " + " | ".join(q["id"] for q in answerable_qs)
          + " | string hits | page hits |\n|---|" + "---|" * (len(answerable_qs) + 2))
for name, rows in retrieval.items():
    md.append(f"| {name} | " + " | ".join(("HIT" if x["string_hit"] else "miss") + f" {x['pages']}" for x in rows)
              + f" | {sum(x['string_hit'] for x in rows)}/{len(rows)} | {sum(x['page_hit'] for x in rows)}/{len(rows)} |")
for label, rows in answers.items():
    md.append(f"\n## Answers — {label}\n\n| Q | answer | answerable | cited pages | answer_ok | cited_ok |\n|---|---|---|---|---|---|")
    for r in rows:
        a = (r["answer"] or r.get("error", "")).replace("|", "/").replace("\n", " ")
        md.append(f"| {r['id']} | {a} | {r.get('answerable')} | {[s['page'] for s in r['sources']]} | {r['answer_ok']} | {r['cited_ok']} |")
md.append("\n## Multi-turn follow-up (expected: 16% currency-neutral)\n")
for label, m in multiturn.items():
    t2 = m["turns"][1]
    md.append(f"- **{label}** — retrieval query: “{t2.get('retrieval_query')}” → {t2['answer']} "
              f"(answer_ok={t2['answer_ok']}, history messages={m['history_messages']})")
(OUT / f"phase3_summary{SUFFIX}.md").write_text("\n".join(md) + "\n")
print(f"\nSaved outputs/phase3_results{SUFFIX}.json and outputs/phase3_summary{SUFFIX}.md")
