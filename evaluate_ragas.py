"""
Score results.jsonl with RAGAS (github.com/explodinggradients/ragas).

Metrics (RAGAS v0.4 "collections" API)
    faithfulness       - share of answer claims supported by the retrieved contexts
    answer_relevancy   - how well the answer addresses the question (LLM writes
                         questions from the answer; cosine vs. the real question)
    context_precision  - rank-weighted precision of the top-k chunks, judged
                         against the reference answer in questions.csv
    relevant_chunks    - plain count of top-k chunks judged relevant (same
                         verdicts as context_precision)

Usage
    python evaluate_ragas.py [--results results.jsonl] [--questions questions.csv]
    python evaluate_ragas.py --only Q01          # quick test on one question
    python evaluate_ragas.py --limit 3           # first 3 questions

Environment variables
    ANTHROPIC_API_KEY    API key (required)
    RAGAS_JUDGE_MODEL    judge model (default: claude-sonnet-5-5)
    RAGAS_JUDGE_EFFORT   judge effort level (default: medium)
    RAGAS_CONCURRENCY    parallel questions (default: 4)

Outputs
    results_with_ragas_scores.jsonl  - every results.jsonl record + scores
    ragas_summary.csv                - one row per question, for threshold analysis
    ragas_category_summary.csv       - averages per question category
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import sys
import typing as t
from pathlib import Path

# Must be set before tokenizers/torch load: parallel tokenizer threads plus
# concurrent embedding calls segfault on macOS.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from pydantic import BaseModel

from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness
from ragas.metrics.collections.context_precision.util import (
    ContextPrecisionInput,
    ContextPrecisionOutput,
)

from rag_pipeline import EMBED_MODEL, REFUSAL_TEXT

ROOT = Path(__file__).resolve().parent

CATEGORIES = [  # (label, first, last) by question number
    ("direct_retrieval", 1, 5),
    ("conceptual", 6, 10),
    ("cross_document", 11, 15),
    ("unanswerable", 16, 20),
]

# Phrases that signal the model declined (fully or partly) rather than answered.
DECLINE_MARKERS = (
    REFUSAL_TEXT.lower().rstrip("."),
    "do not let me",
    "does not contain",
    "do not contain",
    "not provided in",
    "not covered",
    "documents do not",
    "passages do not",
    "context does not",
)

T = t.TypeVar("T", bound=BaseModel)

DEFAULT_JUDGE_MODEL = "claude-sonnet-5-5"
# Models that accept server-side refusal fallbacks (fallbacks="default").
FALLBACK_MODELS = ("claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1")


class SerialHFEmbeddings(HuggingFaceEmbeddings):
    """Local sentence-transformers embeddings run on the event-loop thread.

    RAGAS's HuggingFaceEmbeddings runs async calls in a thread pool, so with
    several questions scored concurrently the same torch model is called from
    multiple threads at once - which segfaults on macOS. Embedding a handful
    of short questions takes milliseconds, so running them inline is fine.
    """

    async def aembed_text(self, text: str, **kwargs: t.Any) -> t.List[float]:
        return self.embed_text(text, **kwargs)

    async def aembed_texts(self, texts: t.List[str], **kwargs: t.Any) -> t.List[t.List[float]]:
        return self.embed_texts(texts, **kwargs)


# ---------------------------------------------------------------------------
# Judge LLM
# ---------------------------------------------------------------------------

class ClaudeJudge(InstructorBaseRagasLLM):
    """RAGAS judge backed by the Anthropic SDK's structured outputs.

    Replaces ragas.llm_factory(provider="anthropic"), which does not work with
    anthropic SDK 1.x / current Claude models: it always sends temperature and
    top_p (TypeError in SDK 1.x, 400 on Opus 4.7+), caps max_tokens at 1024
    (too small once thinking tokens are counted), and relies on instructor's
    forced tool calling (400 on Opus 5.5 / Sonnet 5.5). messages.parse() uses
    native structured outputs instead.
    """

    def __init__(self, model: str, effort: str, max_tokens: int = 16000):
        import anthropic

        self.client = anthropic.AsyncAnthropic(max_retries=5)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    async def agenerate(self, prompt: str, response_model: type[T]) -> T:
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": prompt}],
            output_format=response_model,
        )
        if self.model in FALLBACK_MODELS:
            # If a safety classifier declines, retry server-side on Anthropic's
            # recommended fallback model instead of failing the metric.
            resp = await self.client.beta.messages.parse(
                **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            )
        else:
            resp = await self.client.messages.parse(**kwargs)
        self.calls += 1
        self.input_tokens += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            raise RuntimeError(f"judge returned no parsable output (stop_reason={resp.stop_reason})")
        return resp.parsed_output

    def generate(self, prompt: str, response_model: type[T]) -> T:
        return asyncio.run(self.agenerate(prompt, response_model))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def qnum(query_id: str) -> int:
    return int("".join(ch for ch in query_id if ch.isdigit()))


def category_of(query_id: str) -> str:
    n = qnum(query_id)
    for label, lo, hi in CATEGORIES:
        if lo <= n <= hi:
            return label
    return "other"


def declined(answer: str) -> bool:
    a = (answer or "").lower()
    return any(m in a for m in DECLINE_MARKERS)


def refusal_outcome(rec: dict, answerable: bool) -> str:
    """Classify refusal behaviour against the ground-truth answerability."""
    full = bool(rec.get("refused"))
    partial = not full and declined(rec.get("answer") or "")
    if answerable:
        if full:
            return "false_refusal"
        return "answered_with_caveat" if partial else "answered"
    if full:
        return "correct_refusal"
    return "partial_refusal" if partial else "hallucinated"


def nan_to_none(x: float | None) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), 4)


def mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

async def context_precision_with_verdicts(metric: ContextPrecision, question: str,
                                          reference: str, contexts: list[str]):
    """Same computation as ContextPrecision.ascore, but also returns the
    per-chunk 0/1 verdicts so we can report 'relevant chunks out of k'."""
    verdicts = []
    for ctx in contexts:
        prompt = metric.prompt.to_string(
            ContextPrecisionInput(question=question, context=ctx, answer=reference)
        )
        out = await metric.llm.agenerate(prompt, ContextPrecisionOutput)
        verdicts.append(int(out.verdict))
    return metric._calculate_average_precision(verdicts), verdicts


async def score_one(rec: dict, q: dict, metrics: dict, sem: asyncio.Semaphore) -> dict:
    question, answer, contexts = rec["question"], rec.get("answer") or "", rec["contexts"]
    reference = q["reference_answer"]
    scores: dict = {"ragas_errors": []}

    async with sem:
        async def run(name, coro):
            try:
                return await coro
            except Exception as e:
                scores["ragas_errors"].append(f"{name}: {type(e).__name__}: {e}")
                return None

        faith, rel, cp = await asyncio.gather(
            run("faithfulness", metrics["faithfulness"].ascore(
                user_input=question, response=answer, retrieved_contexts=contexts)),
            run("answer_relevancy", metrics["answer_relevancy"].ascore(
                user_input=question, response=answer)),
            run("context_precision", context_precision_with_verdicts(
                metrics["context_precision"], question, reference, contexts)),
        )

    # MetricResult defines __len__ (raises for float values), so never test it
    # for truthiness - compare with None explicitly.
    scores["faithfulness"] = nan_to_none(faith.value) if faith is not None else None
    scores["answer_relevancy"] = nan_to_none(rel.value) if rel is not None else None
    if cp is not None:
        scores["context_precision"] = nan_to_none(cp[0])
        scores["chunk_relevance_verdicts"] = cp[1]
        scores["relevant_chunks"] = sum(cp[1])
    else:
        scores["context_precision"] = scores["relevant_chunks"] = None
        scores["chunk_relevance_verdicts"] = None
    print(f"  {q['query_id']}: faith={scores['faithfulness']} rel={scores['answer_relevancy']} "
          f"cp={scores['context_precision']} relevant={scores['relevant_chunks']}"
          f"{'  ERRORS' if scores['ragas_errors'] else ''}", flush=True)
    return scores


async def main_async(args) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set.")

    # Join results to questions by question text (batch ids may be 1..N).
    with open(args.questions, newline="", encoding="utf-8-sig") as f:
        questions = {row["question"].strip(): row for row in csv.DictReader(f)}
    records = [json.loads(line) for line in open(args.results, encoding="utf-8")]
    pairs = []
    for rec in records:
        q = questions.get(rec["question"].strip())
        if q is None:
            sys.exit(f"Question not found in {args.questions}: {rec['question'][:80]}")
        if rec.get("error") or rec.get("answer") is None:
            sys.exit(f"{q['query_id']} has no answer (error={rec.get('error')}); re-run the batch first.")
        pairs.append((rec, q))

    if args.only:
        wanted = {s.strip().upper() for s in args.only.split(",")}
        pairs = [p for p in pairs if p[1]["query_id"].upper() in wanted]
        missing = wanted - {p[1]["query_id"].upper() for p in pairs}
        if missing:
            sys.exit(f"Not found in {args.results}: {', '.join(sorted(missing))}")
    if args.limit:
        pairs = pairs[: args.limit]

    judge = ClaudeJudge(args.judge_model, args.judge_effort)
    embeddings = SerialHFEmbeddings(model=EMBED_MODEL)
    metrics = {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings, strictness=3),
        "context_precision": ContextPrecision(llm=judge),
    }

    print(f"Scoring {len(pairs)} answers with judge={args.judge_model} (effort={args.judge_effort}) ...")
    sem = asyncio.Semaphore(args.concurrency)
    all_scores = await asyncio.gather(*(score_one(r, q, metrics, sem) for r, q in pairs))

    # ---- per-question outputs -------------------------------------------
    out_rows = []
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for (rec, q), sc in sorted(zip(pairs, all_scores), key=lambda p: qnum(p[0][1]["query_id"])):
            answerable = q["answerable"].strip() == "1"
            merged = {
                **rec,
                "id": q["query_id"],
                "query_id": q["query_id"],
                "category": category_of(q["query_id"]),
                "answerable": answerable,
                "reference_answer": q["reference_answer"],
                "refusal_outcome": refusal_outcome(rec, answerable),
                **sc,
                "judge_model": args.judge_model,
            }
            f.write(json.dumps(merged, ensure_ascii=False) + "\n")
            out_rows.append(merged)

    summary_cols = [
        "query_id", "category", "answerable", "refused", "refusal_outcome",
        "faithfulness", "answer_relevancy", "context_precision", "relevant_chunks",
        "top1_score", "mean_topk_score", "n_cited", "invalid_citations",
        "docs_retrieved", "output_tokens", "latency_s", "question",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols)
        w.writeheader()
        for r in out_rows:
            w.writerow({
                **{k: r.get(k) for k in summary_cols},
                "answerable": int(r["answerable"]),
                "refused": int(bool(r.get("refused"))),
                "n_cited": len(r.get("cited_passages") or []),
                "invalid_citations": len(r.get("invalid_citations") or []),
                "docs_retrieved": len({s["doc"] for s in r["sources"]}),
            })

    # ---- category summary -------------------------------------------------
    cat_rows = []
    for label, _, _ in CATEGORIES + [("ALL_ANSWERABLE", 0, 0)]:
        rows = ([r for r in out_rows if r["answerable"]] if label == "ALL_ANSWERABLE"
                else [r for r in out_rows if r["category"] == label])
        if not rows:
            continue
        cat_rows.append({
            "category": label,
            "n": len(rows),
            "faithfulness": mean([r["faithfulness"] for r in rows]),
            "answer_relevancy": mean([r["answer_relevancy"] for r in rows]),
            "context_precision": mean([r["context_precision"] for r in rows]),
            "relevant_chunks_of_5": mean([r["relevant_chunks"] for r in rows]),
            "top1_score": mean([r["top1_score"] for r in rows]),
            "mean_topk_score": mean([r["mean_topk_score"] for r in rows]),
            "refusal_rate": round(sum(bool(r["refused"]) for r in rows) / len(rows), 4),
        })
    with open(args.out_cat_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cat_rows[0].keys()))
        w.writeheader()
        w.writerows(cat_rows)

    # ---- console report -------------------------------------------------
    fmt = lambda x: "  -  " if x is None else f"{x:.3f}"
    print("\nPer question")
    print(f"{'id':4s} {'category':17s} {'faith':>6s} {'relev':>6s} {'ctxP':>6s} {'rel/5':>5s} {'top1':>6s}  outcome")
    for r in out_rows:
        print(f"{r['query_id']:4s} {r['category']:17s} {fmt(r['faithfulness']):>6s} "
              f"{fmt(r['answer_relevancy']):>6s} {fmt(r['context_precision']):>6s} "
              f"{str(r['relevant_chunks']):>5s} {r['top1_score']:6.3f}  {r['refusal_outcome']}")

    print("\nBy category")
    print(f"{'category':17s} {'n':>2s} {'faith':>6s} {'relev':>6s} {'ctxP':>6s} {'rel/5':>6s} {'top1':>6s} {'refused':>7s}")
    for c in cat_rows:
        print(f"{c['category']:17s} {c['n']:2d} {fmt(c['faithfulness']):>6s} {fmt(c['answer_relevancy']):>6s} "
              f"{fmt(c['context_precision']):>6s} {fmt(c['relevant_chunks_of_5']):>6s} "
              f"{fmt(c['top1_score']):>6s} {c['refusal_rate']:7.0%}")

    print("\nUnanswerable questions (Q16-Q20)")
    for r in out_rows:
        if not r["answerable"]:
            print(f"  {r['query_id']}: {r['refusal_outcome']:16s} {r['question']}")
    flagged = [r for r in out_rows if r["answerable"] and r["refusal_outcome"] != "answered"]
    if flagged:
        print("\nAnswerable questions that were refused or hedged")
        for r in flagged:
            print(f"  {r['query_id']}: {r['refusal_outcome']:20s} {r['question']}")

    errs = [(r["query_id"], e) for r in out_rows for e in r["ragas_errors"]]
    if errs:
        print(f"\nRAGAS errors ({len(errs)}):")
        for qid, e in errs:
            print(f"  {qid}: {e[:200]}")

    print(f"\nJudge usage: {judge.calls} calls, {judge.input_tokens} input / {judge.output_tokens} output tokens")
    print(f"Wrote {args.out_jsonl}, {args.out_csv}, {args.out_cat_csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results.jsonl"))
    ap.add_argument("--questions", default=str(ROOT / "questions.csv"))
    ap.add_argument("--out-jsonl", default=str(ROOT / "results_with_ragas_scores.jsonl"))
    ap.add_argument("--out-csv", default=str(ROOT / "ragas_summary.csv"))
    ap.add_argument("--out-cat-csv", default=str(ROOT / "ragas_category_summary.csv"))
    ap.add_argument("--judge-model", default=os.environ.get("RAGAS_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    ap.add_argument("--judge-effort", default=os.environ.get("RAGAS_JUDGE_EFFORT", "medium"))
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("RAGAS_CONCURRENCY", "4")))
    ap.add_argument("--only", help="score only these query ids, e.g. Q01 or Q01,Q16")
    ap.add_argument("--limit", type=int, help="score only the first N questions")
    args = ap.parse_args()

    # Test runs (--only/--limit) write *_test files so full results are not overwritten.
    if args.only or args.limit:
        for attr in ("out_jsonl", "out_csv", "out_cat_csv"):
            if getattr(args, attr) == ap.get_default(attr):
                p = Path(getattr(args, attr))
                setattr(args, attr, str(p.with_name(f"{p.stem}_test{p.suffix}")))
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
