#!/usr/bin/env python3
"""REST API for the Multi-Cloud Cost Triage Agent.

Run locally:
    uv run uvicorn api:app --reload

Endpoints
---------
GET  /health         liveness probe
GET  /providers      supported providers and their default models
POST /ask            answer a question (JSON)
POST /ask/stream     answer a question as server-sent events
DELETE /sessions/id  forget a conversation

Label review (deterministic, no LLM call):
GET  /labels/coverage                          untagged / suspect counts
GET  /resources                                review queue (filterable)
GET  /resources/{id}/labels                    suggestions for one resource
PUT  /resources/{id}/labels/{label}/decision   accept / reject / override
DELETE /resources/{id}/labels/{label}/decision undo a decision
GET  /labels/decisions                         all recorded decisions
GET  /labels/export?format=csv|sh|tfvars       resolved changes to apply

Pass the same ``session_id`` on follow-up requests to keep conversation memory.

Environment variables:
    OBS_AGENT_API_KEY  — if set, requests must send it in the X-API-Key header
    OBS_AGENT_DECISIONS — where review decisions are stored (default label_decisions.json)
    plus the usual provider keys (DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, ...)
"""

from __future__ import annotations

import hmac
import json
import os
import threading
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

import label_review
from agent import DEFAULT_PROVIDER, PROVIDER_DEFAULT_MODELS, run, stream_run
from label_review import DecisionError, DecisionStore

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
decisions = DecisionStore()


# ── Schemas ────────────────────────────────────────────────────────────


class AskRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "Which Azure subscription has the highest overage?",
                    "provider": "ollama",
                    "session_id": "demo",
                }
            ]
        }
    }

    question: str = Field(min_length=1, max_length=4000)
    provider: str = Field(default=DEFAULT_PROVIDER)
    model: str | None = Field(
        default=None, description="Provider-specific model; omit for the provider default."
    )
    session_id: str | None = Field(
        default=None,
        description="Reuse across requests to keep conversation memory. "
        "Generated and returned if omitted.",
    )


class AskResponse(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "sub-a1b2c3d4 has the highest overage at $18,411.",
                    "session_id": "demo",
                    "provider": "ollama",
                    "model": "qwen2.5:7b",
                    "tools_called": ["get_azure_top_overages"],
                }
            ]
        }
    }

    answer: str
    session_id: str
    provider: str
    model: str
    tools_called: list[str] = Field(
        description="Tools the agent invoked while answering this question."
    )


class ErrorResponse(BaseModel):
    detail: str


ERROR_RESPONSES = {
    401: {
        "model": ErrorResponse,
        "description": "Missing or invalid X-API-Key (only when OBS_AGENT_API_KEY is set).",
    },
    422: {"description": "Invalid request body or unknown provider."},
    502: {
        "model": ErrorResponse,
        "description": "The LLM backend failed (bad key, no credits, server down).",
    },
}

SSE_DESCRIPTION = """Answer a question as server-sent events (`text/event-stream`).

Events, in order:

```
data: {"token": "The top "}            <- repeated, one per text chunk

event: done
data: {"session_id": "demo", "tools_called": ["get_gcp_top_projects"]}
```

If the backend fails mid-stream the final event is instead:

```
event: error
data: {"detail": "..."}
```

Errors that occur before streaming starts (401, 422) are normal HTTP errors.
"""


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
    last_user = max((i for i, m in enumerate(history) if m.get("role") == "user"), default=-1)
    names: list[str] = []
    for m in history[last_user + 1 :]:
        for call in m.get("tool_calls") or []:
            names.append(call["name"])
    return names


# ── Routes ─────────────────────────────────────────────────────────────


@app.get("/health", summary="Liveness probe")
def health() -> dict:
    return {"status": "ok"}


@app.get(
    "/providers",
    dependencies=[Depends(require_api_key)],
    summary="List providers",
    responses={401: ERROR_RESPONSES[401]},
)
def providers() -> dict:
    return {"default": DEFAULT_PROVIDER, "models": PROVIDER_DEFAULT_MODELS}


@app.post(
    "/ask",
    response_model=AskResponse,
    dependencies=[Depends(require_api_key)],
    summary="Ask a question",
    responses={k: v for k, v in ERROR_RESPONSES.items()},
)
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


@app.post(
    "/ask/stream",
    dependencies=[Depends(require_api_key)],
    summary="Ask a question (streamed)",
    description=SSE_DESCRIPTION,
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "Server-sent events."},
        401: ERROR_RESPONSES[401],
        422: ERROR_RESPONSES[422],
    },
)
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


@app.delete(
    "/sessions/{session_id}",
    dependencies=[Depends(require_api_key)],
    summary="Forget a conversation",
    responses={
        401: ERROR_RESPONSES[401],
        404: {"model": ErrorResponse, "description": "Unknown session."},
    },
)
def delete_session(session_id: str) -> dict:
    if not sessions.delete(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}


# ── Label review ───────────────────────────────────────────────────────

LabelName = Literal["team", "env"]


class Alternative(BaseModel):
    value: str
    confidence: float


class LabelState(BaseModel):
    current: str | None = Field(description="Usable tag after cleaning, or null.")
    raw: str | None = Field(description="Tag value exactly as found on the resource.")
    status: Literal["ok", "untagged", "suspect"]
    suggested: str | None
    confidence: float | None
    action: Literal["auto-apply", "needs review"] | None
    source: str | None = Field(description="rule:parent, rule:name, model or out-of-fold model.")
    reason: str | None
    alternatives: list[Alternative]


class CostCenter(BaseModel):
    current: str | None
    implied: str | None = Field(description="Cost center of the suggested (or current) team.")


class ResourceLabels(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "resource_id": "svc-ingest",
                    "cloud": "azure",
                    "parent": "mg-security",
                    "owners": ["hana"],
                    "needs_attention": True,
                    "cost_center": {"current": None, "implied": "CC-2002"},
                    "labels": {
                        "team": {
                            "current": None,
                            "raw": None,
                            "status": "untagged",
                            "suggested": "data-platform",
                            "confidence": 0.6,
                            "action": "needs review",
                            "source": "model",
                            "reason": "strongest signals: creator=hana (+1.27), ...",
                            "alternatives": [{"value": "observability", "confidence": 0.12}],
                        }
                    },
                }
            ]
        }
    }

    resource_id: str
    cloud: Literal["azure", "gcp"]
    parent: str
    owners: list[str]
    labels: dict[str, LabelState]
    cost_center: CostCenter
    needs_attention: bool


class ResourcePage(BaseModel):
    total: int
    items: list[ResourceLabels]


class LabelCounts(BaseModel):
    untagged: int
    suspect: int
    auto_apply_ready: int


class Coverage(BaseModel):
    total: int
    needs_attention: int
    auto_apply_threshold: float
    labels: dict[str, LabelCounts]


class DecisionRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"decision": "accept"},
                {"decision": "override", "value": "data-platform"},
            ]
        }
    }

    decision: Literal["accept", "reject", "override"]
    value: str | None = Field(default=None, description="Required for override.")


class Decision(BaseModel):
    resource_id: str
    label: str
    decision: Literal["accept", "reject", "override"]
    value: str | None = Field(description="The value that will be applied (null for reject).")
    suggested: str | None
    previous: str | None
    decided_at: str


NOT_FOUND = {404: {"model": ErrorResponse, "description": "Unknown resource."}}


def _report_or_404(resource_id: str) -> dict:
    report = label_review.resource_report(resource_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Unknown resource '{resource_id}'")
    return report


@app.get(
    "/labels/coverage",
    response_model=Coverage,
    dependencies=[Depends(require_api_key)],
    summary="Label coverage",
    tags=["labels"],
    responses={401: ERROR_RESPONSES[401]},
)
def labels_coverage() -> dict:
    return label_review.coverage()


@app.get(
    "/resources",
    response_model=ResourcePage,
    dependencies=[Depends(require_api_key)],
    summary="Review queue",
    description="Resources with an untagged or suspected-wrong label, most confident "
    "suggestions first. Filters combine with AND.",
    tags=["labels"],
    responses={401: ERROR_RESPONSES[401]},
)
def list_resources(
    label: LabelName | None = Query(default=None, description="Only check this label."),
    status: Literal["untagged", "suspect"] | None = Query(default=None),
    cloud: Literal["azure", "gcp"] | None = Query(default=None),
    min_confidence: float = Query(default=0.0, ge=0, le=1),
    max_confidence: float = Query(default=1.0, ge=0, le=1),
    search: str | None = Query(default=None, description="Substring of resource_id."),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict:
    items = label_review.review_queue(
        label=label,
        status=status,
        cloud=cloud,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        search=search,
    )
    return {"total": len(items), "items": items[offset : offset + limit]}


@app.get(
    "/resources/{resource_id}/labels",
    response_model=ResourceLabels,
    dependencies=[Depends(require_api_key)],
    summary="Label suggestions for one resource",
    tags=["labels"],
    responses={401: ERROR_RESPONSES[401], **NOT_FOUND},
)
def resource_labels(resource_id: str) -> dict:
    return _report_or_404(resource_id)


@app.put(
    "/resources/{resource_id}/labels/{label}/decision",
    response_model=Decision,
    dependencies=[Depends(require_api_key)],
    summary="Record a review decision",
    description="`accept` applies the suggestion, `override` applies `value` instead, "
    "`reject` keeps the current tag. Replaces any earlier decision.",
    tags=["labels"],
    responses={
        401: ERROR_RESPONSES[401],
        **NOT_FOUND,
        422: {"model": ErrorResponse, "description": "Invalid value, or nothing to accept."},
    },
)
def put_decision(resource_id: str, label: LabelName, req: DecisionRequest) -> dict:
    _report_or_404(resource_id)
    try:
        return decisions.set(resource_id, label, req.decision, req.value)
    except DecisionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete(
    "/resources/{resource_id}/labels/{label}/decision",
    dependencies=[Depends(require_api_key)],
    summary="Undo a review decision",
    tags=["labels"],
    responses={
        401: ERROR_RESPONSES[401],
        404: {"model": ErrorResponse, "description": "No such decision."},
    },
)
def delete_decision(resource_id: str, label: LabelName) -> dict:
    if not decisions.delete(resource_id, label):
        raise HTTPException(status_code=404, detail="Decision not found")
    return {"deleted": f"{resource_id}/{label}"}


@app.get(
    "/labels/decisions",
    response_model=list[Decision],
    dependencies=[Depends(require_api_key)],
    summary="All review decisions",
    tags=["labels"],
    responses={401: ERROR_RESPONSES[401]},
)
def list_decisions() -> list[dict]:
    return decisions.all()


EXPORT_MEDIA = {"csv": "text/csv", "sh": "text/x-shellscript", "tfvars": "application/json"}
EXPORT_FILENAME = {
    "csv": "label-changes.csv",
    "sh": "apply-labels.sh",
    "tfvars": "labels.auto.tfvars.json",
}


@app.get(
    "/labels/export",
    response_class=PlainTextResponse,
    dependencies=[Depends(require_api_key)],
    summary="Export label changes",
    description="Every resource whose resolved tags differ from today's: reviewer "
    "overrides, then accepts, then (if `include_auto`) suggestions at or "
    "above the auto-apply threshold. `sh` emits `az tag update` / "
    "`gcloud projects update` commands; `tfvars` a map for Terraform.",
    tags=["labels"],
    responses={200: {"content": {m: {} for m in EXPORT_MEDIA.values()}}, 401: ERROR_RESPONSES[401]},
)
def export_labels(
    format: Literal["csv", "sh", "tfvars"] = Query(default="csv"),
    include_auto: bool = Query(default=True),
) -> PlainTextResponse:
    body = label_review.export(decisions, format, include_auto)
    return PlainTextResponse(
        body,
        media_type=EXPORT_MEDIA[format],
        headers={"Content-Disposition": f'attachment; filename="{EXPORT_FILENAME[format]}"'},
    )
