"""
Chat interface for the RAG assistant in rag_pipeline.py (pipeline logic unchanged).

    GET  /          Chat page (static/index.html + static/style.css + static/chat.js).
    GET  /api/docs  Titles of the indexed documents (shown on the empty chat).
    POST /api/ask   JSON {"question": "...", "history": [{"role", "content"}, ...]}
                    -> answer with citations and sources, or a tailored decline.

Follow-up questions: when there is chat history, a short Claude call first rewrites
the latest message into a standalone question ("explain the second one" ->
"Explain the AI Risk Management pillar of Malaysia's AI Code of Ethics"). That
standalone question then goes through the unchanged pipeline (answer_question).

The answer is shown only when the model did not refuse AND top1_score >= 0.80.
Otherwise a second short Claude call writes a decline specific to the question
(what the documents do not cover, a related topic they do cover, and one suggested
question) - it is told never to answer from its own knowledge. A template decline
is used if that call fails.

Chat history lives in the browser only - nothing is stored on the server.
Requires ANTHROPIC_API_KEY in the environment.
Run: uvicorn app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_pipeline import Config, Retriever, accepts_sampling, answer_question, get_embedder

STATIC_DIR = Path(__file__).resolve().parent / "static"
CONFIDENCE_THRESHOLD = 0.80  # from threshold_analysis.py
MAX_HISTORY_TURNS = 6        # messages of context sent for follow-up rewriting
MAX_TURN_CHARS = 1500        # per message, keeps the rewrite prompt small

DOC_TITLES = {
    "AI_Boardroom_Malaysia.pdf": "AI in the Boardroom (Malaysia)",
    "AI_Code_of_Ethics_Malaysia.pdf": "Malaysia AI Code of Ethics",
    "AI_Index_Report_Standford_2026.pdf": "Stanford AI Index Report 2026",
    "National_AI_Action_Plan.pdf": "National AI Action Plan 2026–2030",
}
DOC_LIST = "\n".join(f"- {t}" for t in DOC_TITLES.values())

# The embedding model is not safe to call from several threads at once, and
# FastAPI runs sync endpoints in a thread pool - so pipeline calls take turns.
PIPELINE_LOCK = threading.Lock()
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    cfg = Config()
    STATE["cfg"] = cfg
    STATE["retriever"] = Retriever(cfg)
    STATE["client"] = anthropic.Anthropic()
    get_embedder(cfg.embed_model)  # load once at startup, not on the first request
    yield


app = FastAPI(title="Policy Assistant", lifespan=lifespan)


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[Turn] = []


# ---------------------------------------------------------------------------
# Small Claude helpers (same model as the pipeline)
# ---------------------------------------------------------------------------

class Rewrite(BaseModel):
    standalone_question: str


class Decline(BaseModel):
    message: str
    suggested_question: str  # empty string when nothing fits


def llm_parse(system: str, user: str, output_format: type[BaseModel]) -> BaseModel:
    cfg = STATE["cfg"]
    kwargs: dict = {}
    if accepts_sampling(cfg.model):
        kwargs["extra_body"] = {"temperature": 0}  # SDK 1.x: temperature goes in the body
    else:
        kwargs["output_config"] = {"effort": "low"}
    resp = STATE["client"].messages.parse(
        model=cfg.model, max_tokens=2000, system=system,
        messages=[{"role": "user", "content": user}],
        output_format=output_format, **kwargs,
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise RuntimeError(f"no parsable output (stop_reason={resp.stop_reason})")
    return resp.parsed_output


REWRITE_SYSTEM = f"""You turn the user's latest chat message into a standalone question for a search system over these documents:
{DOC_LIST}

Rules:
- Use the conversation only to resolve references such as "it", "they", "that one", "the second one", "explain more".
- Name the subject explicitly (the document, framework, pillar, plan or topic being discussed) so the question makes sense with no conversation.
- Do not answer the question and do not add facts that are not in the conversation.
- If the latest message is already standalone, or starts a new topic, return it unchanged."""


def rewrite_followup(question: str, history: list[Turn]) -> str:
    convo = "\n".join(
        f"{'User' if t.role == 'user' else 'Assistant'}: {t.content.strip()[:MAX_TURN_CHARS]}"
        for t in history[-MAX_HISTORY_TURNS:]
    )
    out = llm_parse(
        REWRITE_SYSTEM,
        f"<conversation>\n{convo}\n</conversation>\n\nLatest message: {question}",
        Rewrite,
    )
    rewritten = out.standalone_question.strip()
    return rewritten if 0 < len(rewritten) <= 2000 else question


DECLINE_SYSTEM = f"""You write the reply for a document Q&A assistant when it could NOT find a reliable answer to the user's question.
The assistant can only answer from these documents:
{DOC_LIST}

Write "message": 2-3 short, friendly sentences to the user that
- say plainly that this particular question can't be answered from the documents, referring to what they actually asked;
- if the retrieved passages touch on a related topic, mention that topic and which document covers it.

Write "suggested_question": one question the documents CAN answer that is close to what the user wants, based only on the passages; or "" if nothing fits.

Strict rules:
- Never answer the user's question or state facts about its subject from your own knowledge - no names, numbers, dates or claims beyond what the passages say.
- Do not list all the documents. No headings or bullet points."""


def write_decline(question: str, reason: str, rec: dict) -> tuple[str, str]:
    passages = "\n\n".join(
        f"[{s['rank']}] {DOC_TITLES.get(s['doc'], s['doc'])}, p.{s['page']}:\n{text[:400]}"
        for s, text in list(zip(rec["sources"], rec["contexts"]))[:3]
    )
    try:
        out = llm_parse(
            DECLINE_SYSTEM,
            f"User question: {question}\nWhy it could not be answered: {reason}\n\n"
            f"Closest passages found:\n{passages}",
            Decline,
        )
        return out.message.strip(), out.suggested_question.strip()
    except Exception:
        return template_decline(question, rec), ""


def suggestion_passes(suggestion: str) -> bool:
    """Retrieval-only check (no Claude call) that a suggested question clears the threshold."""
    with PIPELINE_LOCK:
        rec = answer_question(suggestion, STATE["retriever"], STATE["cfg"], retrieve_only=True)
    return rec["top1_score"] is not None and rec["top1_score"] >= CONFIDENCE_THRESHOLD


def template_decline(question: str, rec: dict) -> str:
    """Fallback when the decline call fails - still specific to the question."""
    nearest = sorted({DOC_TITLES.get(s["doc"], s["doc"]) for s in rec["sources"][:3]})
    if rec.get("refused"):
        return (f"I looked through {' and '.join(nearest)} but none of it answers "
                f"“{question}”, so I'd rather not guess. "
                "Try asking about something those documents discuss directly.")
    return (f"“{question}” doesn't seem to be covered by the documents I can search, "
            "which focus on Malaysia's AI ethics, governance and national strategy, plus the "
            "Stanford AI Index. Try rephrasing with more specific terms.")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.post("/api/ask")
def api_ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question is empty")
    if len(question) > 2000:
        raise HTTPException(400, "Question is too long (max 2000 characters)")

    standalone = question
    if req.history:
        try:
            standalone = rewrite_followup(question, req.history)
        except Exception:
            standalone = question  # fall back to the literal message
    rewritten = " ".join(standalone.lower().split()) != " ".join(question.lower().split())

    with PIPELINE_LOCK:
        rec = answer_question(standalone, STATE["retriever"], STATE["cfg"])

    base = {"standalone_question": standalone, "rewritten": rewritten}
    if rec.get("error"):
        return {**base, "status": "error",
                "message": "Sorry, something went wrong while answering. Please try again in a moment."}

    top1 = rec["top1_score"]
    if rec.get("refused") or top1 is None or top1 < CONFIDENCE_THRESHOLD:
        reason = ("the documents do not contain this information" if rec.get("refused")
                  else "no passage in the documents closely matched the question")
        message, suggestion = write_decline(standalone, reason, rec)
        if suggestion and not suggestion_passes(suggestion):
            suggestion = ""  # don't offer a question that would itself be declined
        return {**base, "status": "out_of_scope", "message": message,
                "suggestion": suggestion, "top1_score": top1}

    return {
        **base,
        "status": "answered",
        "answer": rec["answer"],
        "top1_score": top1,
        "sources": [
            {"n": s["rank"], "title": DOC_TITLES.get(s["doc"], s["doc"]), "file": s["doc"],
             "page": s["page"], "score": s["score"], "text": text}
            for s, text in zip(rec["sources"], rec["contexts"])
        ],
    }


@app.get("/api/docs")
def api_docs():
    return list(DOC_TITLES.values())


# ---------------------------------------------------------------------------
# Chat page: static/index.html, static/style.css, static/chat.js
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")
