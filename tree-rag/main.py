import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import answer as pipeline
import supabase_client as db
from api_guards import check_answer
from env_api import router as env_router
from retrieval import HybridRetriever

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tree-api")

# At most this many attempts per question. A second attempt is made only when the first answer has a
# serious warning (no citation, mixed answer types, unverified species) and the LLM was used.
MAX_ATTEMPTS = max(1, int(os.getenv("ASK_MAX_ATTEMPTS", "2")))
# Comma separated list of allowed browser origins. "*" is fine for local development only.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
# Set PUBLIC_ASK=0 in .env to require a login for /ask as well (recommended when the API is deployed,
# because every question uses free-tier tokens).
PUBLIC_ASK = os.getenv("PUBLIC_ASK", "1") != "0"

state = {}


@asynccontextmanager
async def lifespan(app):
    log.info("Loading the retriever (this reads the knowledge base from Qdrant) ...")
    retriever = HybridRetriever(regions=pipeline.ALLOWED_REGIONS)
    pipeline.build_name_index(retriever)  # build the species-name index now, not on the first request
    state["retriever"] = retriever
    log.info("Ready: %d chunks, configuration %s, login %s",
             len(retriever.points), pipeline.config_hash(), "on" if db.configured() else "NOT configured")
    yield
    state.clear()


app = FastAPI(
    title="Tree Plantation Evidence API",
    description="Evidence-grounded tree recommendations for Thatta, Sindh. Answers only from the knowledge base.",
    version="0.3.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(env_router)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class ChatRequest(AskRequest):
    conversation_id: Optional[uuid.UUID] = None


class SourceOut(BaseModel):
    number: int
    source: str
    page: int
    score: float


class WarningOut(BaseModel):
    code: str
    severity: str
    message: str


class AskResponse(BaseModel):
    question: str
    answer: str
    answer_type: str          # "answer", "partial" or "unavailable"
    reliable: bool            # False if the answer still has a serious warning
    sources: list[SourceOut]
    warnings: list[WarningOut]
    top_score: float
    used_llm: bool
    attempts: int
    latency_ms: int
    config: str


class ChatResponse(AskResponse):
    conversation_id: str
    saved: bool               # False if the answer could not be stored in the chat history


def api_error(code, message, status=503):
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def current_user(authorization):
    """Check the 'Authorization: Bearer <token>' header with Supabase and return the user."""
    if not db.configured():
        raise api_error("auth_not_configured", "Login is not configured on the server (SUPABASE_URL / SUPABASE_ANON_KEY).")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise api_error("missing_token", "Please sign in first. Send the header 'Authorization: Bearer <token>'.", 401)
    token = authorization[7:].strip()
    try:
        user = db.verify_token(token)
    except db.AuthError:
        raise api_error("invalid_token", "Your session has expired or is not valid. Please sign in again.", 401)
    except requests.RequestException as error:
        log.error("Supabase Auth unreachable: %s", error)
        raise api_error("auth_unreachable", "The login service could not be reached.", 502)
    return {"token": token, **user}


def run_pipeline(question):
    """Run the RAG pipeline with the guard layer. Returns (result, answer_type, warnings, reliable,
    attempts, latency_ms) or raises an HTTPException with a clear message."""
    if "retriever" not in state:
        raise api_error("starting", "The server is still loading. Try again in a moment.")

    retriever = state["retriever"]
    started = time.time()
    result = answer_type = warnings = reliable = None
    attempts = 0
    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            attempts = attempt
            result = pipeline.answer_question(retriever, question)
            answer_type, warnings, reliable = check_answer(result, pipeline.EVIDENCE_THRESHOLD)
            if reliable or not result["used_llm"]:
                break
            log.warning("Attempt %d has serious warnings %s", attempt, [w["code"] for w in warnings])
    except pipeline.DailyLimitError as error:
        log.error("Groq daily limit: %s", error)
        raise api_error("llm_daily_limit", str(error))
    except RuntimeError as error:
        text = str(error)
        log.error("Pipeline error: %s", text[:300])
        if "429" in text:
            raise api_error("llm_busy", "The language model is busy right now. Please try again in a minute.")
        raise api_error("llm_error", "The language model could not answer this question.", 502)
    except requests.HTTPError as error:
        status = error.response.status_code if error.response is not None else None
        log.error("Jina error (status %s)", status)
        if status == 429:
            raise api_error("search_busy", "The search service is busy right now. Please try again in a minute.")
        raise api_error("search_error", "The search service returned an error.", 502)
    except requests.exceptions.Timeout:
        raise api_error("timeout", "A service took too long to respond.", 504)
    except requests.exceptions.ConnectionError:
        raise api_error("connection_error", "Could not reach an external service.", 502)

    latency_ms = int((time.time() - started) * 1000)
    log.info("question | type=%s reliable=%s attempts=%d top=%.3f %dms | %s",
             answer_type, reliable, attempts, result["top_score"], latency_ms, question[:80])
    return result, answer_type, warnings, reliable, attempts, latency_ms


def response_fields(question, result, answer_type, warnings, reliable, attempts, latency_ms):
    return dict(
        question=question,
        answer=result["answer"],
        answer_type=answer_type,
        reliable=reliable,
        sources=[
            SourceOut(number=s["number"], source=s["source"], page=s["page"], score=round(float(s["score"]), 3))
            for s in result["sources"]
        ],
        warnings=[WarningOut(**w) for w in warnings],
        top_score=round(float(result["top_score"]), 3),
        used_llm=result["used_llm"],
        attempts=attempts,
        latency_ms=latency_ms,
        config=pipeline.config_hash(),
    )


def make_title(question):
    title = " ".join(question.split())
    return title if len(title) <= 60 else title[:57].rstrip() + "..."


# Plain "def" (not "async def"): the pipeline makes blocking network calls, so FastAPI runs it in a thread.
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, authorization: Optional[str] = Header(default=None)):
    if not PUBLIC_ASK:
        current_user(authorization)
    question = request.question.strip()
    return AskResponse(**response_fields(question, *run_pipeline(question)))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, authorization: Optional[str] = Header(default=None)):
    user = current_user(authorization)
    token = user["token"]
    question = request.question.strip()
    conversation_id = str(request.conversation_id) if request.conversation_id else None

    # Check the conversation first, so that a wrong id does not use any language-model tokens.
    if conversation_id:
        try:
            if not db.get_conversation(token, conversation_id):
                raise api_error("conversation_not_found", "This conversation does not exist.", 404)
        except (db.DatabaseError, requests.RequestException) as error:
            log.error("Database error: %s", error)
            raise api_error("database_error", "The chat history could not be reached.", 502)

    result, answer_type, warnings, reliable, attempts, latency_ms = run_pipeline(question)
    fields = response_fields(question, result, answer_type, warnings, reliable, attempts, latency_ms)

    saved = True
    try:
        if not conversation_id:
            conversation_id = db.create_conversation(token, make_title(question))["id"]
        db.add_messages(token, [
            {"conversation_id": conversation_id, "role": "user", "content": question},
            {
                "conversation_id": conversation_id,
                "role": "assistant",
                "content": result["answer"],
                "answer_type": answer_type,
                "reliable": reliable,
                "sources": [s.model_dump() for s in fields["sources"]],
                "warnings": [w.model_dump() for w in fields["warnings"]],
                "top_score": fields["top_score"],
                "config": fields["config"],
            },
        ])
    except (db.DatabaseError, requests.RequestException) as error:
        saved = False
        log.error("Could not save the chat: %s", error)

    return ChatResponse(**fields, conversation_id=conversation_id or "", saved=saved)


@app.get("/conversations")
def conversations(authorization: Optional[str] = Header(default=None)):
    user = current_user(authorization)
    try:
        return db.list_conversations(user["token"])
    except (db.DatabaseError, requests.RequestException) as error:
        log.error("Database error: %s", error)
        raise api_error("database_error", "The chat history could not be reached.", 502)


@app.get("/conversations/{conversation_id}/messages")
def conversation_messages(conversation_id: uuid.UUID, authorization: Optional[str] = Header(default=None)):
    user = current_user(authorization)
    token, cid = user["token"], str(conversation_id)
    try:
        if not db.get_conversation(token, cid):
            raise api_error("conversation_not_found", "This conversation does not exist.", 404)
        return db.list_messages(token, cid)
    except (db.DatabaseError, requests.RequestException) as error:
        log.error("Database error: %s", error)
        raise api_error("database_error", "The chat history could not be reached.", 502)


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: uuid.UUID, authorization: Optional[str] = Header(default=None)):
    user = current_user(authorization)
    try:
        db.delete_conversation(user["token"], str(conversation_id))
    except (db.DatabaseError, requests.RequestException) as error:
        log.error("Database error: %s", error)
        raise api_error("database_error", "The chat history could not be reached.", 502)
    return {"deleted": str(conversation_id)}


@app.get("/health")
def health():
    retriever = state.get("retriever")
    return {
        "status": "ok" if retriever else "starting",
        "chunks": len(retriever.points) if retriever else 0,
        "regions": pipeline.ALLOWED_REGIONS,
        "evidence_threshold": pipeline.EVIDENCE_THRESHOLD,
        "model": pipeline.GROQ_MODEL,
        "config": pipeline.config_hash(),
        "login_configured": db.configured(),
        "public_ask": PUBLIC_ASK,
    }