"""
RAG assistant over an internal policy/report corpus.

BAA5123 AI and Decision Making - Topic 18: Retrieval-Augmented Assistants
for Internal Knowledge.

Pipeline
    PDFs (data/) -> pypdf page text -> character chunks (with doc + page)
    -> BAAI/bge-small-en-v1.5 embeddings (normalised) -> FAISS inner-product
    index (= cosine similarity) -> top-k retrieval -> Claude answer grounded
    ONLY in the retrieved chunks, with numbered citations [1], [2], ...

Usage
    python rag_pipeline.py build [--chunk-size 1200] [--chunk-overlap 200]
    python rag_pipeline.py ask "What is ...?" [--top-k 5] [--retrieve-only]
    python rag_pipeline.py batch [--questions questions.csv] [--out results.jsonl]

Environment variables
    ANTHROPIC_API_KEY   API key for the Anthropic API (required for ask/batch)
    ANTHROPIC_MODEL     generator model id (default: claude-haiku-4-5)
    RAG_EFFORT          effort level for models without sampling params
                        (low | medium | high; default: medium)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL = "claude-haiku-4-5"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this instruction on queries (not on passages).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

REFUSAL_TEXT = "I cannot answer this question from the provided documents."

# Newer Claude models reject sampling parameters (temperature/top_p/top_k)
# with a 400 error. For these we omit temperature and control determinism /
# depth with `effort` instead. Older models (e.g. claude-haiku-4-5,
# claude-sonnet-4-6) still accept temperature=0.
SAMPLING_REJECTED_PREFIXES = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos",
    "claude-opus-4-7",
    "claude-opus-4-8",
)
# Models that support server-side refusal fallbacks ("default" routing).
FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")


@dataclass
class Config:
    data_dir: str = str(ROOT / "data")
    index_dir: str = str(ROOT / "index")
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 5
    embed_model: str = EMBED_MODEL
    min_page_chars: int = 50  # pages with less text are reported as extraction problems
    min_chunk_chars: int = 50  # drop tiny tail fragments
    model: str = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    effort: str = os.environ.get("RAG_EFFORT", "medium")
    max_tokens: int = 16000


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

SURROGATES = re.compile(r"[\ud800-\udfff]")


def clean_text(text: str) -> str:
    """Normalise whitespace and re-join words hyphenated across line breaks."""
    # pypdf can emit unpaired UTF-16 surrogates for some math glyphs; the
    # tokenizer rejects them, so drop them.
    text = SURROGATES.sub("", text).replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def load_pdfs(data_dir: str, min_page_chars: int = 50) -> tuple[list[dict], list[dict]]:
    """Read every PDF page by page.

    Returns (pages, problems). Each page is {doc, page, text}; page numbers
    are 1-based physical page numbers in the PDF.
    """
    from pypdf import PdfReader

    pages, problems = [], []
    pdf_paths = sorted(Path(data_dir).glob("*.pdf"))
    if not pdf_paths:
        sys.exit(f"No PDFs found in {data_dir}")

    for path in pdf_paths:
        try:
            reader = PdfReader(str(path))
        except Exception as e:  # corrupt / encrypted file
            problems.append({"doc": path.name, "page": None, "issue": f"cannot open: {e}", "skipped": True})
            continue
        for i, page in enumerate(reader.pages, start=1):
            try:
                raw = page.extract_text() or ""
            except Exception as e:
                problems.append({"doc": path.name, "page": i, "issue": f"extract error: {e}", "skipped": True})
                continue
            n_bad = len(SURROGATES.findall(raw))
            if n_bad:
                problems.append({
                    "doc": path.name, "page": i,
                    "issue": f"{n_bad} invalid unicode surrogate char(s) removed (garbled math glyph) - page still indexed",
                    "skipped": False,
                })
            text = clean_text(raw)
            if len(text) < min_page_chars:
                problems.append({
                    "doc": path.name, "page": i,
                    "issue": f"little/no extractable text ({len(text)} chars) - likely image-only or blank page",
                    "skipped": True,
                })
                continue
            pages.append({"doc": path.name, "page": i, "text": text})
    return pages, problems


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Sliding character window; ends are snapped back to whitespace so
    words are not split (only if that keeps at least 80% of the window)."""
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = text.rfind(" ", start + int(size * 0.8), end)
            cut = max(cut, text.rfind("\n", start + int(size * 0.8), end))
            if cut > start:
                end = cut
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = end - overlap
    return chunks


def chunk_pages(pages: list[dict], cfg: Config) -> list[dict]:
    """Chunk within each page so every chunk keeps an exact page citation."""
    chunks = []
    for p in pages:
        for j, piece in enumerate(chunk_text(p["text"], cfg.chunk_size, cfg.chunk_overlap)):
            if len(piece) < cfg.min_chunk_chars:
                continue
            chunks.append({
                "chunk_id": f"{p['doc']}::p{p['page']}::c{j}",
                "doc": p["doc"],
                "page": p["page"],
                "text": piece,
            })
    return chunks


# ---------------------------------------------------------------------------
# 3. Embedding + FAISS index
# ---------------------------------------------------------------------------

_EMBEDDER = None


def get_embedder(name: str):
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(name)
    return _EMBEDDER


def embed(texts: list[str], cfg: Config, is_query: bool = False) -> np.ndarray:
    model = get_embedder(cfg.embed_model)
    if is_query:
        texts = [BGE_QUERY_INSTRUCTION + t for t in texts]
    vecs = model.encode(
        texts, batch_size=64, normalize_embeddings=True,
        show_progress_bar=not is_query, convert_to_numpy=True,
    )
    return vecs.astype("float32")


def build_index(cfg: Config) -> None:
    import faiss

    t0 = time.time()
    pages, problems = load_pdfs(cfg.data_dir, cfg.min_page_chars)
    chunks = chunk_pages(pages, cfg)
    if not chunks:
        sys.exit("No chunks produced - nothing to index.")

    print(f"Embedding {len(chunks)} chunks with {cfg.embed_model} ...")
    vecs = embed([c["text"] for c in chunks], cfg)

    # Vectors are L2-normalised, so inner product == cosine similarity.
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    out = Path(cfg.index_dir)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "faiss.index"))
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Per-document stats
    stats: dict[str, dict] = {}
    for p in pages:
        s = stats.setdefault(p["doc"], {"pages_with_text": 0, "chunks": 0, "chars": 0, "pages_skipped": 0})
        s["pages_with_text"] += 1
        s["chars"] += len(p["text"])
    for c in chunks:
        stats[c["doc"]]["chunks"] += 1
    for pr in problems:
        stats.setdefault(pr["doc"], {"pages_with_text": 0, "chunks": 0, "chars": 0, "pages_skipped": 0})
        stats[pr["doc"]]["pages_skipped"] += pr["skipped"]

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embed_model": cfg.embed_model,
        "dim": int(vecs.shape[1]),
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "min_page_chars": cfg.min_page_chars,
        "n_chunks": len(chunks),
        "per_document": stats,
        "extraction_problems": problems,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Report
    print(f"\nIndex written to {out}/ in {time.time() - t0:.1f}s")
    print(f"Config: chunk_size={cfg.chunk_size}, chunk_overlap={cfg.chunk_overlap}, "
          f"embed_model={cfg.embed_model}, dim={vecs.shape[1]}, metric=cosine (IndexFlatIP)")
    print(f"\n{'Document':45s} {'pages':>6s} {'skipped':>8s} {'chars':>9s} {'chunks':>7s}")
    for doc, s in sorted(stats.items()):
        print(f"{doc:45s} {s['pages_with_text']:6d} {s['pages_skipped']:8d} {s['chars']:9d} {s['chunks']:7d}")
    print(f"{'TOTAL':45s} {sum(s['pages_with_text'] for s in stats.values()):6d} "
          f"{sum(p['skipped'] for p in problems):8d} {sum(s['chars'] for s in stats.values()):9d} {len(chunks):7d}")
    if problems:
        print(f"\nExtraction problems ({len(problems)}):")
        for pr in problems:
            print(f"  - {pr['doc']} p.{pr['page']}: {pr['issue']}")


# ---------------------------------------------------------------------------
# 4. Retrieval
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self, cfg: Config):
        import faiss

        idx_path = Path(cfg.index_dir) / "faiss.index"
        if not idx_path.exists():
            sys.exit("Index not found - run `python rag_pipeline.py build` first.")
        self.cfg = cfg
        self.index = faiss.read_index(str(idx_path))
        with open(Path(cfg.index_dir) / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.meta = json.loads((Path(cfg.index_dir) / "meta.json").read_text())
        # Embed queries with the same model the index was built with.
        cfg.embed_model = self.meta.get("embed_model", cfg.embed_model)

    def search(self, question: str, k: int) -> list[dict]:
        q = embed([question], self.cfg, is_query=True)
        scores, ids = self.index.search(q, k)
        hits = []
        for rank, (i, s) in enumerate(zip(ids[0], scores[0]), start=1):
            if i < 0:
                continue
            hits.append({**self.chunks[i], "rank": rank, "score": round(float(s), 4)})
        return hits


# ---------------------------------------------------------------------------
# 5. Generation (Anthropic API)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are an internal knowledge assistant. You answer questions using ONLY the numbered context passages supplied in the user message.

Rules:
1. Use only information stated in the context passages. Do not use outside knowledge, prior training data, or assumptions, even if you know the answer.
2. Support every factual statement with a citation to the passage number(s) it came from, in square brackets, e.g. [1] or [2][4]. Only cite passage numbers that exist in the context.
3. If the context does not contain enough information to answer, reply with exactly: "{REFUSAL_TEXT}" and nothing else.
4. If the context only partially answers the question, answer the supported part with citations and state clearly what the documents do not cover.
5. Be concise and factual. Do not invent numbers, dates, names, or document titles."""


def format_context(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{n}] (source: {h['doc']}, page {h['page']})\n{h['text']}"
        for n, h in enumerate(hits, start=1)
    )


def accepts_sampling(model: str) -> bool:
    return not model.startswith(SAMPLING_REJECTED_PREFIXES)


def generate(question: str, hits: list[dict], cfg: Config) -> dict:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    user_msg = (
        f"Context passages:\n\n{format_context(hits)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context passages above, with numbered citations."
    )
    kwargs: dict = {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_msg}],
    }
    temperature = None
    effort = None
    if accepts_sampling(cfg.model):
        # anthropic SDK 1.x dropped the `temperature` keyword; it is still
        # honoured by these models when sent in the request body.
        temperature = 0.0
        kwargs["extra_body"] = {"temperature": temperature}
    else:
        effort = cfg.effort
        kwargs["output_config"] = {"effort": effort}

    t0 = time.time()
    if cfg.model in FALLBACK_MODELS:
        # If a safety classifier declines, retry server-side on Anthropic's
        # recommended fallback model instead of returning a refusal.
        resp = client.beta.messages.create(
            **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
    else:
        resp = client.messages.create(**kwargs)
    latency = time.time() - t0

    answer = "".join(b.text for b in resp.content if b.type == "text").strip()
    return {
        "answer": answer,
        "stop_reason": resp.stop_reason,
        "served_by": resp.model,
        "temperature": temperature,
        "effort": effort,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "latency_s": round(latency, 2),
    }


# ---------------------------------------------------------------------------
# 6. End-to-end query + logging
# ---------------------------------------------------------------------------

def is_refusal(answer: str, stop_reason: str | None) -> bool:
    if stop_reason == "refusal":
        return True
    return REFUSAL_TEXT.lower().rstrip(".") in answer.lower()


def answer_question(question: str, retriever: Retriever, cfg: Config,
                    retrieve_only: bool = False) -> dict:
    hits = retriever.search(question, cfg.top_k)
    scores = [h["score"] for h in hits]
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "answer": None,
        "contexts": [h["text"] for h in hits],
        "sources": [
            {"rank": h["rank"], "doc": h["doc"], "page": h["page"],
             "chunk_id": h["chunk_id"], "score": h["score"]}
            for h in hits
        ],
        "retrieval_scores": scores,
        "top1_score": scores[0] if scores else None,
        "mean_topk_score": round(float(np.mean(scores)), 4) if scores else None,
        "refused": None,
        "config": {
            "model": cfg.model, "top_k": cfg.top_k,
            "chunk_size": retriever.meta["chunk_size"],
            "chunk_overlap": retriever.meta["chunk_overlap"],
            "embed_model": retriever.meta["embed_model"],
        },
    }
    if retrieve_only:
        return record

    try:
        gen = generate(question, hits, cfg)
    except Exception as e:  # keep batch runs going; the error is logged
        record["error"] = f"{type(e).__name__}: {e}"
        return record

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", gen["answer"])})
    record.update({
        "answer": gen["answer"],
        "refused": is_refusal(gen["answer"], gen["stop_reason"]),
        "cited_passages": cited,
        "invalid_citations": [n for n in cited if not 1 <= n <= len(hits)],
        "stop_reason": gen["stop_reason"],
        "served_by": gen["served_by"],
        "temperature": gen["temperature"],
        "effort": gen["effort"],
        "input_tokens": gen["input_tokens"],
        "output_tokens": gen["output_tokens"],
        "latency_s": gen["latency_s"],
    })
    return record


def append_jsonl(path: str | Path, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_record(r: dict) -> None:
    print("=" * 80)
    print(f"Q: {r['question']}")
    print("-" * 80)
    if r.get("error"):
        print(f"ERROR: {r['error']}")
    elif r["answer"] is not None:
        print(r["answer"])
    print("-" * 80)
    print("Sources:")
    for s in r["sources"]:
        print(f"  [{s['rank']}] {s['doc']} p.{s['page']}  (cosine={s['score']:.4f})")
    print(f"top1_score={r['top1_score']:.4f}  mean_topk_score={r['mean_topk_score']:.4f}  "
          f"refused={r['refused']}")
    if r.get("stop_reason"):
        print(f"model={r['served_by']}  temperature={r['temperature']}  effort={r['effort']}  "
              f"tokens in/out={r['input_tokens']}/{r['output_tokens']}  latency={r['latency_s']}s")


def run_batch(questions_csv: str, out_path: str, retriever: Retriever, cfg: Config,
              retrieve_only: bool = False) -> None:
    with open(questions_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or "question" not in rows[0]:
        sys.exit(f"{questions_csv} must have a 'question' column")

    Path(out_path).unlink(missing_ok=True)  # each batch run starts a fresh file
    n_ref = n_err = 0
    for i, row in enumerate(rows, start=1):
        q = row["question"].strip()
        if not q:
            continue
        rec = answer_question(q, retriever, cfg, retrieve_only)
        rec["id"] = row.get("id") or row.get("query_id") or str(i)
        append_jsonl(out_path, rec)
        n_ref += bool(rec.get("refused"))
        n_err += "error" in rec
        print(f"[{i}/{len(rows)}] top1={rec['top1_score']:.3f} refused={rec['refused']} "
              f"{'ERROR ' if 'error' in rec else ''}{q[:70]}")
    print(f"\nWrote {len(rows)} records to {out_path} (refused={n_ref}, errors={n_err})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="RAG assistant over the PDF corpus in data/")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="extract, chunk, embed and index the PDFs")
    b.add_argument("--chunk-size", type=int, default=Config.chunk_size)
    b.add_argument("--chunk-overlap", type=int, default=Config.chunk_overlap)
    b.add_argument("--data-dir", default=Config.data_dir)
    b.add_argument("--index-dir", default=Config.index_dir)

    a = sub.add_parser("ask", help="answer one question")
    a.add_argument("question")
    a.add_argument("--log", default=str(ROOT / "logs" / "ask_log.jsonl"))

    bt = sub.add_parser("batch", help="answer every question in a CSV")
    bt.add_argument("--questions", default=str(ROOT / "questions.csv"))
    bt.add_argument("--out", default=str(ROOT / "results.jsonl"))

    for p in (a, bt):
        p.add_argument("--top-k", type=int, default=Config.top_k)
        p.add_argument("--index-dir", default=Config.index_dir)
        p.add_argument("--retrieve-only", action="store_true",
                       help="skip the LLM call (no API key needed)")

    args = ap.parse_args()
    cfg = Config(index_dir=args.index_dir)

    if args.cmd == "build":
        cfg.chunk_size, cfg.chunk_overlap, cfg.data_dir = args.chunk_size, args.chunk_overlap, args.data_dir
        build_index(cfg)
        return

    cfg.top_k = args.top_k
    if not args.retrieve_only and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set (use --retrieve-only to test retrieval without it).")
    retriever = Retriever(cfg)

    if args.cmd == "ask":
        rec = answer_question(args.question, retriever, cfg, args.retrieve_only)
        append_jsonl(args.log, rec)
        print_record(rec)
    else:
        run_batch(args.questions, args.out, retriever, cfg, args.retrieve_only)


if __name__ == "__main__":
    main()
