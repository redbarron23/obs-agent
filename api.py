#!/usr/bin/env python3
"""REST API for the Multi-Cloud Cost Triage Agent.

Run locally:
    uvicorn api:app --reload

Endpoints
---------
GET  /health         liveness probe
GET  /providers      supported providers and their default models
POST /ask            answer a question (JSON)
POST /ask/stream     answer a question as server-sent events
DELETE /sessions/id  forget a conversation

Pass the same ``session_id`` on follow-up requests to keep conversation memory.

Environment variables:
    OBS_AGENT_API_KEY  — if set, requests must send it in the X-API-Key header
    plus the usual provider keys (DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, ...)
"""

from __future__ import annotations

import hmac
import json
import os
import threading
import uuid
from collections import OrderedDict
from typing import Iterator

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent import DEFAULT_PROVIDER, PROVIDER_DEFAULT_MODELS, run, stream_run

MAX_SESSIONS = 200

app = FastAPI(
    title="obs-agent API",
    description="Chat-based cloud logging cost triage across Azure and GCP.",
    version="0.1.0",
)


# ── Auth ───────────────────────────────────────────────────────────────

def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Enforce X-API-Key when OBS_AGENT_API_KEY is configured."""
    expected = os.environ.get("OBS_AGENT_API_KEY")
    if not expected:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Session memory ─────────────────────────────────────────────────────

class SessionStore:
    """Thread-safe in-memory conversation store with LRU eviction."""

    def __init__(self, max_sessions: int = MAX_SESSIONS):
        self._max = max_sessions
        self._data: OrderedDict[str, list[dict]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: str) -> list[dict] | None:
        with self._lock:
            history = self._data.get(session_id)
            if history is not None:
                self._data.move_to_end(session_id)
            return list(history) if history is not None else None

    def put(self, session_id: str, history: list[dict]) -> None:
        with self._lock:
            self._data[session_id] = history
            self._data.move_to_end(session_id)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._data.pop(session_id, None) is not None


sessions = SessionStore()


# ── Schemas ────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    provider: str = Field(default=DEFAULT_PROVIDER)
    model: str | None = None
    session_id: str | None = Field(
        default=None, description="Reuse to keep conversation memory."
    )


class AskResponse(BaseModel):
    answer: str
    session_id: str
    provider: str
    model: str
    tools_called: list[str]


def _validate(req: AskRequest) -> str:
    if req.provider not in PROVIDER_DEFAULT_MODELS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown provider '{req.provider}'. "
                   f"Choose from {sorted(PROVIDER_DEFAULT_MODELS)}.",
        )
    return req.model or PROVIDER_DEFAULT_MODELS[req.provider]


def _tools_called(history: list[dict]) -> list[str]:
    """Names of tools invoked during the conversation's latest turn."""
    last_user = max(
        (i for i, m in enumerate(history) if m.get("role") == "user"), default=-1
    )
    names: list[str] = []
    for m in history[last_user + 1:]:
        for call in m.get("tool_calls") or []:
            names.append(call["name"])
    return names


# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/providers", dependencies=[Depends(require_api_key)])
def providers() -> dict:
    return {"default": DEFAULT_PROVIDER, "models": PROVIDER_DEFAULT_MODELS}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask(req: AskRequest) -> AskResponse:
    model = _validate(req)
    session_id = req.session_id or uuid.uuid4().hex
    try:
        answer, history = run(
            req.question,
            provider=req.provider,
            model=model,
            messages=sessions.get(session_id),
        )
    except Exception as exc:  # provider/auth/network failures
        raise HTTPException(status_code=502, detail=f"LLM backend error: {exc}") from exc
    sessions.put(session_id, history)
    return AskResponse(
        answer=answer,
        session_id=session_id,
        provider=req.provider,
        model=model,
        tools_called=_tools_called(history),
    )


@app.post("/ask/stream", dependencies=[Depends(require_api_key)])
def ask_stream(req: AskRequest) -> StreamingResponse:
    model = _validate(req)
    session_id = req.session_id or uuid.uuid4().hex

    def events() -> Iterator[str]:
        holder: dict = {}
        try:
            for token in stream_run(
                req.question,
                provider=req.provider,
                model=model,
                messages=sessions.get(session_id),
                result_holder=holder,
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"
            return
        if "messages" not in holder:
            yield f"event: error\ndata: {json.dumps({'detail': 'LLM backend error'})}\n\n"
            return
        sessions.put(session_id, holder["messages"])
        done = {"session_id": session_id, "tools_called": _tools_called(holder["messages"])}
        yield f"event: done\ndata: {json.dumps(done)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.delete("/sessions/{session_id}", dependencies=[Depends(require_api_key)])
def delete_session(session_id: str) -> dict:
    if not sessions.delete(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}
