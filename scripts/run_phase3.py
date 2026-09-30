"""Phase 3 evaluation: retrieval comparison + LLM answers + grounding checks + multi-turn test.

  export GOOGLE_API_KEY=...                       # Gemini key (never commit it)
  export LLM_MODEL="google_genai:gemini-3.5-flash"  # optional; this is the default
  python scripts/run_phase3.py              # re-run after a rate-limit stop: it resumes, answered questions are kept
  python scripts/run_phase3.py --fresh      # ignore saved answers and ask everything again

Writes outputs/phase3_results.json and outputs/phase3_summary.md (dry run: *_dryrun.*), and saves
after every answer, so a stop (rate limit, Ctrl-C) never loses the answers already paid for.

Free-tier friendly: ~12 Gemini calls by default, one every --delay seconds (default 13 s = under
5 requests/minute). On a 429 it waits the delay Google asks for and retries; if the DAILY quota is
used up it stops asking, saves, and tells you to wait or switch model.
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
ap.add_argument("--delay", type=float, default=13.0, help="seconds between LLM calls (13 s keeps under 5 requests/minute)")
ap.add_argument("--fresh", action="store_true", help="ignore answers saved by an earlier run")
ap.add_argument("--with-baseline", action="store_true",
                help="also generate answers with the Phase 2 retriever (7 more LLM calls)")
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
    llm = get_llm(max_retries=1); model = model_name()  # retries handled in ask() below
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


class QuotaExhausted(Exception):
    pass


def _retry_seconds(msg: str) -> float | None:
    import re
    m = re.search(r"retry in ([0-9.]+)s", msg) or re.search(r"retryDelay['\"]?: ?['\"]?([0-9.]+)s", msg)
    return float(m.group(1)) if m else None


def ask(chat, question, session):
    """One question; waits and retries on per-minute rate limits; raises QuotaExhausted on daily/zero quota."""
    t = time.time()
    for attempt in range(4):
        try:
            out = chat.invoke({"question": question}, config={"configurable": {"session_id": session}})
            break
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            rate = "429" in msg or "RESOURCE_EXHAUSTED" in msg or "RateLimit" in msg
            if rate and ("PerDay" in msg or "per day" in msg.lower() or "limit: 0" in msg):
                raise QuotaExhausted(msg[:600])
            if rate and attempt < 3:
                wait = (_retry_seconds(msg) or 30.0) + 3
                print(f"  rate limited (attempt {attempt + 1}); waiting {wait:.0f}s ...", flush=True)
                time.sleep(wait)
                continue
            out = {"answer": "", "answerable": None, "sources": [], "error": msg[:500]}  # recorded, not hidden
            break
    out["latency_s"] = round(time.time() - t, 1)
    time.sleep(args.delay)
    return out


SUFFIX = "_dryrun" if args.dry_run else ""
RES_PATH = OUT / f"phase3_results{SUFFIX}.json"
prev = {}
if RES_PATH.exists() and not args.fresh:
    old = json.loads(RES_PATH.read_text())
    if old.get("model") == model:
        prev = old
        print("Resuming: answers saved by an earlier run with this model are reused (use --fresh to redo).")

SYSTEMS = {"final (Phase 3: hybrid@6)": "hybrid@6"}
if args.with_baseline:
    SYSTEMS = {"baseline (Phase 2: semantic@4)": "semantic@4", **SYSTEMS}
answers = {label: [] for label in SYSTEMS}
multiturn = {}
stopped = None


def save():
    res = {"model": model, "dry_run": args.dry_run, "semantic_backend": SEM, "chunks": len(chunks),
           "pages": len(pages), "retrieval": retrieval, "answers": answers, "multiturn": multiturn,
           "complete": stopped is None, "stopped_reason": stopped,
           "generated": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    RES_PATH.write_text(json.dumps(res, indent=2))
    md = [f"# Phase 3 results\n\nModel: `{model}` · semantic backend: `{SEM}` · {len(pages)} pages, "
          f"{len(chunks)} chunks · generated {res['generated']}\n"]
    if args.dry_run:
        md.append("> **DRY RUN — placeholder answers from a scripted fake model. Not results.**\n")
    if stopped:
        md.append(f"> **INCOMPLETE — stopped: {stopped[:300]}** Re-run the script later to finish (answers so far are kept).\n")
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
        md.append(f"- **{label}** — retrieval query: “{t2.get('retrieval_query')}” → {t2['answer'] or t2.get('error')} "
                  f"(answer_ok={t2['answer_ok']}, history messages={m['history_messages']})")
    (OUT / f"phase3_summary{SUFFIX}.md").write_text("\n".join(md) + "\n")


FOLLOW = [("What were NIKE, Inc.'s total revenues in fiscal 2023?", None),
          ("How much did that grow on a currency-neutral basis?", ["16%", "16 percent"])]
try:
    for label, rname in SYSTEMS.items():
        chat = with_memory(build_chain(retrievers[rname], llm))
        print(f"\n=== Answers: {label} ===")
        done = {r["id"]: r for r in prev.get("answers", {}).get(label, []) if not r.get("error")}
        for q in qs:
            if q["id"] in done:
                answers[label].append(done[q["id"]]); print(f"{q['id']}: reused saved answer"); continue
            out = ask(chat, q["question"], f"{rname}-{q['id']}")  # fresh session per question
            ok, cok = check(q, out)
            answers[label].append({"id": q["id"], "question": q["question"], "ground_truth": q["ground_truth"],
                                   **out, "answer_ok": ok, "cited_ok": cok})
            save()
            print(f"\n{q['id']} {q['question']}\n  answer    : {out['answer']}\n"
                  f"  answerable: {out.get('answerable')} | cited pages: {[s['page'] for s in out['sources']]}"
                  f" | retrieved pages: {out.get('retrieved_pages')}\n"
                  f"  auto-check: answer_ok={ok} cited_ok={cok} | {out['latency_s']}s"
                  + (f"\n  ERROR: {out['error']}" if out.get("error") else ""), flush=True)

    # ------------------------------------------------------------ 4. multi-turn (memory) test
    for label, condense in [("without question rewriting (Phase 2)", False), ("with question rewriting (Phase 3)", True)]:
        old = prev.get("multiturn", {}).get(label)
        if old and not any(t.get("error") for t in old["turns"]):
            multiturn[label] = old; print(f"\nMulti-turn, {label}: reused saved result"); continue
        chat = with_memory(build_chain(retrievers["hybrid@6"], llm, condense=condense))
        sid = f"mt-{condense}"; _store.pop(sid, None)
        turns = []
        for q, expect in FOLLOW:
            out = ask(chat, q, sid)
            out["answer_ok"] = None if expect is None else any(e in out.get("answer", "") for e in expect)
            turns.append({"question": q, **out})
        multiturn[label] = {"turns": turns, "history_messages": len(_store[sid].messages)}
        save()
        t2 = turns[1]
        print(f"\n=== Multi-turn, {label} ===\n  turn-2 retrieval query: {t2.get('retrieval_query')}\n"
              f"  turn-2 answer: {t2['answer'] or t2.get('error')}\n  answer_ok (16%): {t2['answer_ok']} | "
              f"history messages: {multiturn[label]['history_messages']}", flush=True)
except QuotaExhausted as e:
    stopped = ("Gemini quota exhausted for this key/model (daily limit, or no free quota for this model). "
               "Wait for the quota to reset, or set LLM_MODEL to another free model, then re-run. " + str(e))
    print("\n" + "!" * 80 + f"\nSTOPPED: {stopped}\n" + "!" * 80)
except KeyboardInterrupt:
    stopped = "interrupted by user"
finally:
    save()
    print(f"\nSaved outputs/phase3_results{SUFFIX}.json and outputs/phase3_summary{SUFFIX}.md"
          + ("" if stopped is None else "  (INCOMPLETE - re-run to finish; saved answers are reused)"))
