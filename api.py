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
DELETE /resources/{id}/labels/{label}/decision revoke a decision (recorded in feedback)
GET  /labels/decisions                         all recorded decisions
GET  /labels/export?format=csv|sh|tfvars       resolved changes to apply
GET  /labels/backlog                           pending review work and its age
GET  /labels/collaboration[/history]           assignments, notes, unresolved questions
GET  /resources/{id}/labels/{label}/collaboration  one label's collaboration state and history
PUT / DELETE /resources/{id}/labels/{label}/assignment   assign / unassign a reviewer
POST /resources/{id}/labels/{label}/notes | /questions | /questions/{qid}/answer | /resolve

Saved private reviews (same functions as the UI's Import / Label review pages;
disabled when OBS_AGENT_DEMO=1; an unknown review ID is a 404, never synthetic data):
POST /reviews                                  validate and save an inventory CSV
GET  /reviews                                  list saved reviews
GET  /reviews/{review_id}                      one review's summary and approval context
GET  /reviews/{review_id}/coverage | /resources | /resources/{id}/labels
PUT / DELETE /reviews/{review_id}/resources/{id}/labels/{label}/decision
GET  /reviews/{review_id}/decisions | /feedback | /feedback/metrics | /backlog
GET  /reviews/{review_id}/export?format=csv|sh|tfvars|plan
GET  /reviews/{review_id}/collaboration[/history] and the same per-label collaboration routes

Operations (see docs/operations.md):
GET  /ops/metrics[?format=prometheus][&review_id=ID]  workflow counters, durations, backlog

Every response carries an X-Request-ID header (echoed when the client sends a
safe one); structured server logs carry the same ID as correlation_id.

Pass the same ``session_id`` on follow-up requests to keep conversation memory.

Environment variables:
    OBS_AGENT_API_KEY  — required; requests must send it in the X-API-Key header
    OBS_AGENT_DECISIONS — SQLite database for review decisions (default label_decisions.db)
    OBS_AGENT_PROVIDER / OBS_AGENT_MODEL — default provider and model for /ask
    plus the usual provider keys (DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, ...)
"""

from __future__ import annotations

import hmac
import json
import os
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Iterator
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

import label_review
import observability
import workflow_metrics
from agent import (
    PROVIDER_DEFAULT_MODELS,
    PROVIDERS,
    ConfigError,
    check_config,
    default_model,
    default_provider,
    model_warning,
    run,
    served_by,
    stream_run,
)
from inventory_import import InventoryImportError
from label_config import LabelConfigError
from label_review import DecisionError, DecisionStore
from review_workspaces import (
    PrivateReviewsDisabled,
    UnknownReviewError,
    import_review,
    is_review_id,
    load_review,
    require_private_reviews,
    review_summary,
    saved_reviews,
)

MAX_SESSIONS = 200
MAX_AGENT_REQUESTS_PER_MINUTE = 20
MAX_CONCURRENT_AGENT_REQUESTS = 4

# Fail at startup, not on the first request, if the defaults are misconfigured.
try:
    check_config()
except ConfigError as exc:
    raise RuntimeError(f"obs-agent API: {exc}") from exc

app = FastAPI(
    title="obs-agent API",
    description="Human-reviewed Azure/GCP label changes, with optional synthetic cost Q&A.",
    version="0.3.0",
)


# ── Correlation IDs ────────────────────────────────────────────────────


@app.middleware("http")
async def correlate(request: Request, call_next):
    """Bind one correlation ID per request; return it as X-Request-ID."""
    correlation_id = workflow_metrics.clean_correlation_id(request.headers.get("x-request-id"))
    tokens = workflow_metrics.bind(correlation_id, surface="api")
    try:
        response = await call_next(request)
    except Exception as exc:
        # Log the type and route template only; the message may quote request data.
        route = request.scope.get("route")
        workflow_metrics.log_unhandled(
            correlation_id, getattr(route, "path", "unknown"), request.method, exc
        )
        response = JSONResponse(
            {"detail": "Internal server error", "correlation_id": correlation_id},
            status_code=500,
        )
    finally:
        workflow_metrics.unbind(tokens)
    response.headers["X-Request-ID"] = correlation_id
    return response


# ── Auth ───────────────────────────────────────────────────────────────


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Enforce the configured X-API-Key on every non-health route."""
    expected = os.environ.get("OBS_AGENT_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="API authentication is not configured")
    if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class AgentRequestLimiter:
    """Small in-process guard against costly concurrent or bursty LLM calls."""

    def __init__(self, per_minute: int, concurrent: int):
        self._per_minute = per_minute
        self._recent: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(concurrent)

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        with self._lock:
            recent = self._recent.setdefault(client, deque())
            while recent and recent[0] <= now - 60:
                recent.popleft()
            if len(recent) >= self._per_minute:
                return False
            recent.append(now)
            return True

    def acquire(self, client: str) -> bool:
        return self.allow(client) and self._slots.acquire(blocking=False)

    def release(self) -> None:
        self._slots.release()


agent_limiter = AgentRequestLimiter(MAX_AGENT_REQUESTS_PER_MINUTE, MAX_CONCURRENT_AGENT_REQUESTS)


def limit_agent_requests(request: Request):
    """Rate-limit and cap concurrent /ask requests per directly connected client."""
    client = request.client.host if request.client else "unknown"
    if not agent_limiter.acquire(client):
        raise HTTPException(
            status_code=429,
            detail="Too many agent requests",
            headers={"Retry-After": "60"},
        )
    try:
        yield
    finally:
        agent_limiter.release()


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
    provider: str | None = Field(
        default=None,
        description="Omit to use the server default (OBS_AGENT_PROVIDER, else ollama).",
    )
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
                    "answer": "sub-a1b2c3d4 has the highest overage at $20,109.04 (USD, 30 days).",
                    "session_id": "demo",
                    "provider": "ollama",
                    "model": "qwen2.5:7b",
                    "served_by": None,
                    "tools_called": ["get_azure_top_overages"],
                    "warning": "qwen2.5:7b is a small local model: ...",
                }
            ]
        }
    }

    answer: str
    session_id: str
    provider: str
    model: str = Field(description="Model that was requested.")
    served_by: str | None = Field(
        default=None,
        description="Model that actually answered, when the provider reports it "
        "(differs from `model` after a refusal fallback).",
    )
    tools_called: list[str] = Field(
        description="Tools the agent invoked while answering this question."
    )
    warning: str | None = Field(
        default=None, description="Caveat about answer reliability, e.g. a small local model."
    )


class ErrorResponse(BaseModel):
    detail: str


ERROR_RESPONSES = {
    401: {
        "model": ErrorResponse,
        "description": "Missing or invalid X-API-Key (only when OBS_AGENT_API_KEY is set).",
    },
    422: {"description": "Invalid request body or unknown provider."},
    429: {"model": ErrorResponse, "description": "Too many agent requests."},
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
data: {"session_id": "demo", "provider": "openai", "model": "gpt-5.4-mini",
       "served_by": "gpt-5.4-mini", "tools_called": ["get_gcp_top_projects"],
       "warning": null}
```

If the backend fails mid-stream the final event is instead:

```
event: error
data: {"detail": "..."}
```

Errors that occur before streaming starts (401, 422) are normal HTTP errors.
"""


def _validate(req: AskRequest) -> tuple[str, str]:
    """(provider, model) for this request, falling back to the server defaults."""
    provider = req.provider or default_provider()
    if provider not in PROVIDERS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown provider '{provider}'. Choose from {sorted(PROVIDERS)}.",
        )
    return provider, req.model or default_model(provider)


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
    provider, model = check_config()
    return {
        "default": provider,
        "default_model": model,
        "models": PROVIDER_DEFAULT_MODELS,
        "warning": model_warning(provider, model),
    }


@app.post(
    "/ask",
    response_model=AskResponse,
    dependencies=[Depends(require_api_key), Depends(limit_agent_requests)],
    summary="Ask a question",
    responses={k: v for k, v in ERROR_RESPONSES.items()},
)
def ask(req: AskRequest) -> AskResponse:
    provider, model = _validate(req)
    session_id = req.session_id or uuid.uuid4().hex
    try:
        answer, history = run(
            req.question,
            provider=provider,
            model=model,
            messages=sessions.get(session_id),
        )
    except Exception as exc:  # provider/auth/network failures
        raise HTTPException(status_code=502, detail=f"LLM backend error: {exc}") from exc
    sessions.put(session_id, history)
    return AskResponse(
        answer=answer,
        session_id=session_id,
        provider=provider,
        model=model,
        served_by=served_by(history),
        tools_called=_tools_called(history),
        warning=model_warning(provider, model),
    )


@app.post(
    "/ask/stream",
    dependencies=[Depends(require_api_key), Depends(limit_agent_requests)],
    summary="Ask a question (streamed)",
    description=SSE_DESCRIPTION,
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "Server-sent events."},
        401: ERROR_RESPONSES[401],
        422: ERROR_RESPONSES[422],
    },
)
def ask_stream(req: AskRequest) -> StreamingResponse:
    provider, model = _validate(req)
    session_id = req.session_id or uuid.uuid4().hex

    def events() -> Iterator[str]:
        holder: dict = {}
        try:
            for token in stream_run(
                req.question,
                provider=provider,
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
        done = {
            "session_id": session_id,
            "provider": provider,
            "model": model,
            "served_by": served_by(holder["messages"]),
            "tools_called": _tools_called(holder["messages"]),
            "warning": model_warning(provider, model),
        }
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


# ── Saved private reviews (shared helpers) ────────────────────────────

REVIEW_ID_PATTERN = r"^[0-9a-f]{64}$"


def private_reviews_enabled() -> None:
    """Saved private reviews are off in the public demo, as on the Import page."""
    try:
        require_private_reviews()
    except PrivateReviewsDisabled as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _open_review(review_id: str):
    """(inventory, decision store) of a saved review; 404 for an unknown ID, never synthetic."""
    private_reviews_enabled()
    if not is_review_id(review_id):  # the route pattern normally rejects it first
        raise HTTPException(status_code=422, detail="Invalid saved review ID")
    try:
        return load_review(review_id)
    except UnknownReviewError as exc:
        raise HTTPException(status_code=404, detail="Saved review not found") from exc


# ── Label review ───────────────────────────────────────────────────────

LabelName = Literal["team", "env"]


class Alternative(BaseModel):
    value: str
    confidence: float


class ConflictingSignal(BaseModel):
    feature: str
    points_to: str
    count: int = Field(description="Labelled resources with this feature and that value.")
    support: int = Field(description="Labelled resources with this feature.")


class WithheldSuggestion(BaseModel):
    value: str
    model_score: float


class EvidenceState(BaseModel):
    status: Literal["supported", "missing_signals", "insufficient_support", "conflicting"]
    reason: str
    supporting: list[str] = Field(default_factory=list)
    conflicting: list[ConflictingSignal] = Field(default_factory=list)
    withheld: WithheldSuggestion | None = Field(
        default=None,
        description="Value the model would have proposed; context only, it cannot be accepted.",
    )


class LabelState(BaseModel):
    current: str | None = Field(description="Usable tag after cleaning, or null.")
    raw: str | None = Field(description="Tag value exactly as found on the resource.")
    status: Literal["ok", "untagged", "suspect"]
    suggested: str | None = Field(
        description="Null when the label is ok or inference abstained (insufficient evidence)."
    )
    confidence: float | None = Field(
        description="Model or rule score for the suggestion; not a calibrated probability."
    )
    action: Literal["auto-apply", "needs review", "insufficient evidence"] | None
    source: str | None = Field(
        description="rule:parent, rule:name, model, out-of-fold model, or abstain."
    )
    reason: str | None
    alternatives: list[Alternative]
    evidence: EvidenceState | None = Field(
        default=None, description="Ownership-evidence check for untagged labels."
    )


class CostCenter(BaseModel):
    current: str | None
    implied: str | None = Field(description="Cost center of the suggested (or current) team.")
    conflicting: list[str] = Field(default_factory=list)


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


class LabelCounts(BaseModel):
    untagged: int
    suspect: int
    auto_apply_ready: int
    insufficient_evidence: int = Field(
        default=0, description="Untagged labels with no suggestion (inference abstained)."
    )
    insufficient_evidence_reasons: dict[str, int] = Field(default_factory=dict)


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
    reviewer: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Reviewer name or ID. Required unless OBS_AGENT_REVIEWER is configured.",
    )
    rationale: str | None = Field(
        default=None,
        max_length=label_review.MAX_NOTE,
        description="Optional reason for the decision (e.g. why the suggestion was overridden). "
        "Kept in the collaboration history only; never in plans, exports, logs or metrics.",
    )


class Decision(BaseModel):
    resource_id: str
    label: str
    decision: Literal["accept", "reject", "override"]
    value: str | None = Field(description="The value that will be applied (null for reject).")
    suggested: str | None
    previous: str | None
    decided_at: str
    reviewer: str | None = None
    inventory_id: str | None = None
    model_version: str | None = None


class RevocationRequest(BaseModel):
    model_config = {"json_schema_extra": {"examples": [{"reviewer": "alice", "reason": "wrong"}]}}

    reviewer: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Reviewer name or ID. Required unless OBS_AGENT_REVIEWER is configured. "
        "Self-reported, like the reviewer on a decision.",
    )
    reason: str | None = Field(
        default=None,
        max_length=label_review.MAX_REVOCATION_REASON,
        description="Optional reason, kept in the review history only (never logs or metrics).",
    )


class ReviewFeedback(Decision):
    id: int
    decision: Literal["accept", "reject", "override", "revoke"]
    confidence: float | None
    source: str | None
    reason: str | None = Field(description="The model's explanation shown to the reviewer.")
    revoked_decision: Literal["accept", "reject", "override"] | None = Field(
        default=None, description="On a `revoke` event: the decision that was withdrawn."
    )
    revoked_reviewer: str | None = None
    revoked_decided_at: str | None = None
    revoked_feedback_id: int | None = Field(
        default=None, description="Feedback row that recorded the revoked decision, if any."
    )
    revocation_reason: str | None = None


class Revocation(BaseModel):
    deleted: str = Field(description="`resource_id/label` of the revoked decision.")
    revocation: ReviewFeedback


class CollaborationSummary(BaseModel):
    assignee: str | None
    notes: int
    unresolved_questions: int


class QueueItem(ResourceLabels):
    collaboration: dict[str, CollaborationSummary] = Field(
        default_factory=dict,
        description="Per label: assignee and counts of notes and unresolved questions. "
        "Collaboration metadata never changes suggestions or approvals.",
    )


class QueuePage(BaseModel):
    total: int
    items: list[QueueItem]


ReviewerId = Field(
    default=None,
    min_length=1,
    max_length=128,
    description="Your reviewer name or ID. Required unless OBS_AGENT_REVIEWER is configured. "
    "Self-reported, like the reviewer on a decision.",
)


class ReviewerRequest(BaseModel):
    reviewer: str | None = ReviewerId


class AssignmentRequest(ReviewerRequest):
    model_config = {"json_schema_extra": {"examples": [{"assignee": "bob", "reviewer": "alice"}]}}

    assignee: str = Field(min_length=1, max_length=128, description="Reviewer ID to assign.")


class NoteRequest(ReviewerRequest):
    model_config = {
        "json_schema_extra": {"examples": [{"text": "Ask the data team", "reviewer": "alice"}]}
    }

    text: str = Field(min_length=1, max_length=label_review.MAX_NOTE)


class ResolveRequest(ReviewerRequest):
    text: str | None = Field(
        default=None, max_length=label_review.MAX_NOTE, description="Optional resolution note."
    )


class CollaborationEvent(BaseModel):
    id: int
    resource_id: str
    label: str
    event: Literal["assign", "unassign", "note", "rationale", "question", "answer", "resolve"]
    reviewer: str = Field(description="Who recorded the event (self-reported).")
    created_at: str
    assignee: str | None = Field(
        default=None, description="assign: the new assignee; unassign: the one removed."
    )
    question_id: int | None = Field(default=None, description="answer / resolve: the question.")
    feedback_id: int | None = Field(
        default=None, description="rationale: the review_feedback row of the decision explained."
    )
    text: str | None = None


class Note(BaseModel):
    id: int
    kind: Literal["note", "rationale"]
    text: str
    feedback_id: int | None
    reviewer: str
    created_at: str


class Answer(BaseModel):
    text: str
    reviewer: str
    created_at: str


class Question(BaseModel):
    id: int
    text: str
    status: Literal["open", "answered", "resolved"]
    asked_by: str
    asked_at: str
    answers: list[Answer]
    resolved_by: str | None
    resolved_at: str | None
    resolution: str | None


class CollaborationItem(BaseModel):
    resource_id: str
    label: str
    assignee: str | None
    assigned_by: str | None
    assigned_at: str | None
    notes: list[Note]
    questions: list[Question]
    unresolved_questions: int = Field(description="Questions that are open or answered.")


class CollaborationDetail(CollaborationItem):
    history: list[CollaborationEvent]


NOT_FOUND = {404: {"model": ErrorResponse, "description": "Unknown resource."}}
COLLABORATION_DESCRIPTION = (
    "Collaboration metadata (assignment, notes, unresolved questions) is history for reviewers. "
    "It is never an approval: it does not change suggestions, decisions, the backlog's reviewed "
    "items, reviewed change plans or exports. Text is never written to logs or metrics."
)
COLLABORATION_ERRORS = {
    401: ERROR_RESPONSES[401],
    404: {"model": ErrorResponse, "description": "Unknown resource, question or assignment."},
    422: {"model": ErrorResponse, "description": "Missing reviewer, invalid text or state."},
}


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
    response_model=QueuePage,
    dependencies=[Depends(require_api_key)],
    summary="Review queue",
    description="Resources with an untagged or suspected-wrong label, highest model score "
    "first; labels withheld for insufficient evidence have no score and come last. "
    "Filters combine with AND.",
    tags=["labels"],
    responses={401: ERROR_RESPONSES[401]},
)
def list_resources(
    label: LabelName | None = Query(default=None, description="Only check this label."),
    assignee: str | None = Query(
        default=None, max_length=128, description="Only labels assigned to this reviewer ID."
    ),
    unresolved: bool | None = Query(
        default=None, description="true: only labels with an unresolved question; false: none."
    ),
    status: Literal["untagged", "suspect"] | None = Query(default=None),
    cloud: Literal["azure", "gcp"] | None = Query(default=None),
    min_confidence: float = Query(default=0.0, ge=0, le=1),
    max_confidence: float = Query(default=1.0, ge=0, le=1),
    search: str | None = Query(default=None, description="Substring of resource_id."),
    evidence: Literal[
        "insufficient", "supported", "missing_signals", "insufficient_support", "conflicting"
    ]
    | None = Query(
        default=None,
        description="Evidence status; 'insufficient' matches every abstained label.",
    ),
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
        evidence=evidence,
        store=decisions,
        assignee=assignee,
        unresolved=unresolved,
    )
    page = label_review.with_collaboration(items[offset : offset + limit], decisions)
    return {"total": len(items), "items": page}


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
        with workflow_metrics.operation("decision_write", invalid=(DecisionError,)):
            return decisions.set(
                resource_id,
                label,
                req.decision,
                req.value,
                reviewer=req.reviewer,
                rationale=req.rationale,
            )
    except DecisionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete(
    "/resources/{resource_id}/labels/{label}/decision",
    response_model=Revocation,
    dependencies=[Depends(require_api_key)],
    summary="Revoke a review decision",
    description="Removes the active decision and appends a `revoke` event (reviewer, time, "
    "optional reason, and the withdrawn decision) to `/labels/feedback` in one transaction. The "
    "optional JSON body supplies `reviewer` and `reason`. Revoking a missing decision returns 404 "
    "and records nothing.",
    tags=["labels"],
    responses={
        401: ERROR_RESPONSES[401],
        404: {"model": ErrorResponse, "description": "No such decision; nothing recorded."},
        422: {"model": ErrorResponse, "description": "Missing reviewer or invalid reason."},
    },
)
def delete_decision(
    resource_id: str, label: LabelName, req: RevocationRequest | None = None
) -> dict:
    return _revoke(decisions, resource_id, label, req)


def _revoke(
    store: DecisionStore, resource_id: str, label: str, req: RevocationRequest | None
) -> dict:
    req = req or RevocationRequest()
    try:
        with workflow_metrics.operation("decision_write", invalid=(DecisionError,)):
            event = store.revoke(resource_id, label, reviewer=req.reviewer, reason=req.reason)
    except DecisionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if event is None:
        raise HTTPException(status_code=404, detail="Decision not found")
    return {"deleted": f"{resource_id}/{label}", "revocation": event}


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


@app.get(
    "/labels/feedback",
    response_model=list[ReviewFeedback],
    dependencies=[Depends(require_api_key)],
    summary="Reviewer feedback with model context",
    description="Append-only review actions, including the suggestion, confidence and explanation "
    "shown when each decision was made. `revoke` events record a withdrawn decision; they are "
    "history, not active approvals.",
    tags=["labels"],
    responses={401: ERROR_RESPONSES[401]},
)
def list_feedback() -> list[dict]:
    return decisions.feedback()


@app.get(
    "/labels/feedback/metrics",
    dependencies=[Depends(require_api_key)],
    summary="Reviewer-agreement metrics over time",
    description="Macro precision, recall, and F1 of suggestions against reviewer-resolved labels, "
    "grouped by label and calendar month. Untagged rejections are reported as unresolved because "
    "they supply no replacement truth.",
    tags=["labels"],
    responses={401: ERROR_RESPONSES[401]},
)
def feedback_metric_report() -> dict:
    return label_review.feedback_metrics(decisions.feedback())


EXPORT_MEDIA = {
    "csv": "text/csv",
    "sh": "text/x-shellscript",
    "tfvars": "application/json",
    "plan": "application/json",
}
EXPORT_FILENAME = {
    "csv": "label-changes.csv",
    "sh": "apply-labels.sh",
    "tfvars": "labels.auto.tfvars.json",
    "plan": "approved-label-change-plan.json",
}


@app.get(
    "/labels/export",
    response_class=PlainTextResponse,
    dependencies=[Depends(require_api_key)],
    summary="Export label changes",
    description="Every resource whose resolved tags differ from today's: reviewer "
    "overrides, then accepts, then (if `include_auto`) suggestions at or "
    "above the auto-apply threshold. `plan` is always reviewed-only and includes "
    "the approval evidence; `sh` emits `az tag update` / `gcloud projects update` "
    "commands; `tfvars` a map for Terraform.",
    tags=["labels"],
    responses={
        200: {"content": {m: {} for m in EXPORT_MEDIA.values()}},
        401: ERROR_RESPONSES[401],
        422: {"model": ErrorResponse, "description": "Invalid destination tag or label value."},
    },
)
def export_labels(
    format: Literal["csv", "sh", "tfvars", "plan"] = Query(default="csv"),
    include_auto: bool = Query(default=False),
) -> PlainTextResponse:
    return _export_response(decisions, format, include_auto)


def _export_response(store: DecisionStore, fmt: str, include_auto: bool, inventory=None):
    try:
        with workflow_metrics.operation("export", invalid=(DecisionError,)):
            body = label_review.export(store, fmt, include_auto, inventory)
    except DecisionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PlainTextResponse(
        body,
        media_type=EXPORT_MEDIA[fmt],
        headers={"Content-Disposition": f'attachment; filename="{EXPORT_FILENAME[fmt]}"'},
    )


@app.get(
    "/labels/backlog",
    dependencies=[Depends(require_api_key)],
    summary="Pending review work",
    description="Untagged or suspect labels with no decision for the current snapshot, and how "
    "long review has been open. `feedback_actions` is the historical action count, not backlog. "
    "Definitions: docs/operations.md.",
    tags=["labels"],
    responses={401: ERROR_RESPONSES[401]},
)
def review_backlog() -> dict:
    return workflow_metrics.backlog(decisions)


@app.get(
    "/ops/metrics",
    dependencies=[Depends(require_api_key)],
    summary="Workflow metrics",
    description="In-process counters and durations for import, inference, decision writes and "
    "exports, plus the review backlog (the synthetic store's, or a saved review's with "
    "`review_id`). `format=prometheus` returns the text exposition format.",
    tags=["operations"],
    responses={401: ERROR_RESPONSES[401]},
)
def ops_metrics(
    format: Literal["json", "prometheus"] = Query(default="json"),
    review_id: str | None = Query(
        default=None,
        pattern=REVIEW_ID_PATTERN,
        description="Report the backlog of this saved review instead of the synthetic store.",
    ),
):
    store, inventory = decisions, None
    if review_id is not None:
        inventory, store = _open_review(review_id)
    try:
        backlog = workflow_metrics.backlog(store, inventory)
    except Exception as exc:  # metrics must stay readable while the workflow is failing
        workflow_metrics.log_unhandled(
            workflow_metrics.current_correlation_id(), "/ops/metrics", "GET", exc
        )
        backlog = None
    if format == "prometheus":
        return PlainTextResponse(
            workflow_metrics.prometheus(backlog), media_type="text/plain; version=0.0.4"
        )
    return workflow_metrics.snapshot() | {"backlog": backlog, "tracing": observability.status()}


# ── Saved private reviews ──────────────────────────────────────────────
#
# The same workspace, validation, inference, decision and export functions as the
# Import and Label review pages, addressed by review ID. Every route needs the API key,
# is refused in demo mode, and returns 404 for an unknown ID (never synthetic data).

MAX_IMPORT_CHARS = 50_000_000

ReviewId = Annotated[
    str,
    PathParam(
        pattern=REVIEW_ID_PATTERN,
        description="Saved review ID (64 lowercase hex digits) from `POST /reviews` or "
        "`GET /reviews`.",
    ),
]
REVIEW_DEPENDENCIES = [Depends(require_api_key), Depends(private_reviews_enabled)]
REVIEW_ERRORS = {
    401: ERROR_RESPONSES[401],
    403: {"model": ErrorResponse, "description": "Private reviews are disabled (demo mode)."},
    404: {"model": ErrorResponse, "description": "Unknown saved review or resource."},
    422: {"model": ErrorResponse, "description": "Invalid review ID or request."},
}


class SavedReview(BaseModel):
    id: str
    name: str
    created_at: str


class SavedReviewSummary(SavedReview):
    resources: int
    inventory_id: str = Field(description="Snapshot fingerprint that approvals are bound to.")
    model_version: str = Field(description="Vocabulary and inference fingerprint.")


class ImportRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "estate-2026-10.csv",
                    "csv": "cloud,resource_id,parent,owners,...\nazure,az-demo-001,...\n",
                    "vocabulary": {"team": {"values": ["payments", "data-platform"]}},
                }
            ]
        }
    }

    name: str = Field(min_length=1, max_length=200, description="Display name, e.g. file name.")
    csv: str = Field(
        min_length=1,
        max_length=MAX_IMPORT_CHARS,
        description="Sanitized inventory CSV text (the Import page's schema).",
    )
    vocabulary: dict | None = Field(
        default=None, description="Team/env vocabulary JSON; omit for the built-in one."
    )
    column_mapping: dict[str, str | None] | None = Field(
        default=None,
        description="Import field -> source column, when the CSV's headers differ from the "
        "schema (as on the Import page's mapping step).",
    )


def saved_review(review_id: ReviewId):
    return _open_review(review_id)


def _saved_report_or_404(resource_id: str, inventory) -> dict:
    report = label_review.resource_report(resource_id, inventory)
    if report is None:
        raise HTTPException(status_code=404, detail="Unknown resource in this saved review")
    return report


@app.post(
    "/reviews",
    response_model=SavedReviewSummary,
    dependencies=REVIEW_DEPENDENCIES,
    summary="Import an inventory as a saved review",
    description="Validates and saves the CSV and vocabulary exactly as the Import page does. "
    "The same snapshot and vocabulary always yield the same review ID (re-submitting reopens it "
    "with its decisions); changed data or vocabulary create a separate review. No LLM or cloud "
    "credentials are used.",
    tags=["reviews"],
    responses=REVIEW_ERRORS,
)
def create_review(req: ImportRequest) -> dict:
    vocabulary = None if req.vocabulary is None else json.dumps(req.vocabulary).encode()
    try:
        return import_review(req.csv.encode(), req.name, vocabulary, req.column_mapping)
    except (InventoryImportError, LabelConfigError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get(
    "/reviews",
    response_model=list[SavedReview],
    dependencies=REVIEW_DEPENDENCIES,
    summary="List saved reviews",
    description="Newest first. Every caller with the API key sees every saved review: "
    "one trusted-team workspace, not tenant-isolated storage.",
    tags=["reviews"],
    responses=REVIEW_ERRORS,
)
def list_saved_reviews() -> list[dict]:
    return saved_reviews()


@app.get(
    "/reviews/{review_id}",
    response_model=SavedReviewSummary,
    dependencies=REVIEW_DEPENDENCIES,
    summary="One saved review",
    tags=["reviews"],
    responses=REVIEW_ERRORS,
)
def get_saved_review(review_id: ReviewId, review=Depends(saved_review)) -> dict:
    return review_summary(review_id, review[0])


@app.get(
    "/reviews/{review_id}/coverage",
    response_model=Coverage,
    dependencies=REVIEW_DEPENDENCIES,
    summary="Label coverage of a saved review",
    tags=["reviews"],
    responses=REVIEW_ERRORS,
)
def saved_review_coverage(review=Depends(saved_review)) -> dict:
    return label_review.coverage(review[0])


@app.get(
    "/reviews/{review_id}/resources",
    response_model=QueuePage,
    dependencies=REVIEW_DEPENDENCIES,
    summary="Review queue of a saved review",
    description="Same filters and order as `GET /resources`, on the saved snapshot.",
    tags=["reviews"],
    responses=REVIEW_ERRORS,
)
def saved_review_queue(
    review=Depends(saved_review),
    label: LabelName | None = Query(default=None),
    assignee: str | None = Query(default=None, max_length=128),
    unresolved: bool | None = Query(default=None),
    status: Literal["untagged", "suspect"] | None = Query(default=None),
    cloud: Literal["azure", "gcp"] | None = Query(default=None),
    min_confidence: float = Query(default=0.0, ge=0, le=1),
    max_confidence: float = Query(default=1.0, ge=0, le=1),
    search: str | None = Query(default=None),
    evidence: Literal[
        "insufficient", "supported", "missing_signals", "insufficient_support", "conflicting"
    ]
    | None = Query(default=None),
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
        inventory=review[0],
        evidence=evidence,
        store=review[1],
        assignee=assignee,
        unresolved=unresolved,
    )
    page = label_review.with_collaboration(items[offset : offset + limit], review[1])
    return {"total": len(items), "items": page}


@app.get(
    "/reviews/{review_id}/resources/{resource_id}/labels",
    response_model=ResourceLabels,
    dependencies=REVIEW_DEPENDENCIES,
    summary="Label suggestions for one resource of a saved review",
    tags=["reviews"],
    responses=REVIEW_ERRORS,
)
def saved_resource_labels(resource_id: str, review=Depends(saved_review)) -> dict:
    return _saved_report_or_404(resource_id, review[0])


@app.put(
    "/reviews/{review_id}/resources/{resource_id}/labels/{label}/decision",
    response_model=Decision,
    dependencies=REVIEW_DEPENDENCIES,
    summary="Record a review decision in a saved review",
    description="As `PUT /resources/{id}/labels/{label}/decision`, bound to the saved snapshot. "
    "A label withheld for insufficient evidence can only be overridden.",
    tags=["reviews"],
    responses=REVIEW_ERRORS,
)
def put_saved_decision(
    resource_id: str, label: LabelName, req: DecisionRequest, review=Depends(saved_review)
) -> dict:
    inventory, store = review
    _saved_report_or_404(resource_id, inventory)
    try:
        with workflow_metrics.operation("decision_write", invalid=(DecisionError,)):
            return store.set(
                resource_id,
                label,
                req.decision,
                req.value,
                inventory,
                reviewer=req.reviewer,
                rationale=req.rationale,
            )
    except DecisionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete(
    "/reviews/{review_id}/resources/{resource_id}/labels/{label}/decision",
    response_model=Revocation,
    dependencies=REVIEW_DEPENDENCIES,
    summary="Revoke a review decision in a saved review",
    description="As `DELETE /resources/{id}/labels/{label}/decision`: removes the active "
    "decision and appends a `revoke` event to this review's feedback. The optional JSON body "
    "supplies `reviewer` and `reason`. A missing decision returns 404 and records nothing.",
    tags=["reviews"],
    responses=REVIEW_ERRORS,
)
def delete_saved_decision(
    resource_id: str,
    label: LabelName,
    req: RevocationRequest | None = None,
    review=Depends(saved_review),
) -> dict:
    return _revoke(review[1], resource_id, label, req)


@app.get(
    "/reviews/{review_id}/decisions",
    response_model=list[Decision],
    dependencies=REVIEW_DEPENDENCIES,
    summary="Decisions of a saved review",
    tags=["reviews"],
    responses=REVIEW_ERRORS,
)
def saved_decisions(
    review=Depends(saved_review),
    current: bool = Query(
        default=False,
        description="Only decisions bound to this snapshot and inference version (exportable).",
    ),
) -> list[dict]:
    inventory, store = review
    return label_review.current_decisions(store, inventory) if current else store.all()


@app.get(
    "/reviews/{review_id}/feedback",
    response_model=list[ReviewFeedback],
    dependencies=REVIEW_DEPENDENCIES,
    summary="Reviewer feedback of a saved review",
    tags=["reviews"],
    responses=REVIEW_ERRORS,
)
def saved_feedback(review=Depends(saved_review)) -> list[dict]:
    return review[1].feedback()


@app.get(
    "/reviews/{review_id}/feedback/metrics",
    dependencies=REVIEW_DEPENDENCIES,
    summary="Reviewer-agreement metrics of a saved review",
    tags=["reviews"],
    responses=REVIEW_ERRORS,
)
def saved_feedback_metrics(review=Depends(saved_review)) -> dict:
    return label_review.feedback_metrics(review[1].feedback())


@app.get(
    "/reviews/{review_id}/export",
    response_class=PlainTextResponse,
    dependencies=REVIEW_DEPENDENCIES,
    summary="Export a saved review's label changes",
    description="As `GET /labels/export`, for the saved snapshot. `plan` is the same reviewed "
    "change plan as the Label review page's download; approvals bound to another snapshot or "
    "inference version are excluded.",
    tags=["reviews"],
    responses={200: {"content": {m: {} for m in EXPORT_MEDIA.values()}}, **REVIEW_ERRORS},
)
def export_saved_review(
    review=Depends(saved_review),
    format: Literal["csv", "sh", "tfvars", "plan"] = Query(default="csv"),
    include_auto: bool = Query(default=False),
) -> PlainTextResponse:
    inventory, store = review
    return _export_response(store, format, include_auto, inventory)


@app.get(
    "/reviews/{review_id}/backlog",
    dependencies=REVIEW_DEPENDENCIES,
    summary="Pending review work in a saved review",
    description="As `GET /labels/backlog`, for the saved snapshot (docs/operations.md).",
    tags=["reviews"],
    responses=REVIEW_ERRORS,
)
def saved_review_backlog(review=Depends(saved_review)) -> dict:
    inventory, store = review
    return workflow_metrics.backlog(store, inventory)


# ── Collaboration: assignments, notes, unresolved questions (#46) ──────
#
# Registered once for the synthetic store and once per saved review, so both call the same
# DecisionStore methods as the review page and the CLI. Nothing here is an approval.


def synthetic_review():
    """(inventory, store) of the synthetic demo, looked up per request."""
    return None, decisions


def _collaboration_write(store: DecisionStore, write) -> object:
    try:
        with workflow_metrics.operation("collaboration_write", invalid=(DecisionError, KeyError)):
            return write()
    except DecisionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown resource or question") from exc


def _register_collaboration(prefix: str, list_path: str, source, dependencies, tag: str) -> None:
    item = prefix + "/resources/{resource_id}/labels/{label}"
    common = {"dependencies": dependencies, "tags": [tag], "responses": COLLABORATION_ERRORS}

    def detail(review, resource_id: str, label: str) -> dict:
        return review[1].collaboration_item(resource_id, label)

    @app.get(
        list_path,
        response_model=list[CollaborationItem],
        summary="Assignments, notes and questions",
        description="Current collaboration state per resource label, filterable by assignee and "
        "unresolved questions. " + COLLABORATION_DESCRIPTION,
        **common,
    )
    def list_collaboration(
        review=Depends(source),
        assignee: str | None = Query(default=None, max_length=128),
        unresolved: bool | None = Query(default=None),
    ) -> list[dict]:
        return review[1].collaboration(assignee=assignee, unresolved=unresolved)

    @app.get(
        list_path + "/history",
        response_model=list[CollaborationEvent],
        summary="Collaboration history",
        description="Append-only, attributable assignment, note, rationale and question events, "
        "oldest first. Separate from `/feedback`, which holds decisions only.",
        **common,
    )
    def collaboration_history(review=Depends(source)) -> list[dict]:
        return review[1].collaboration_history()

    @app.get(
        item + "/collaboration",
        response_model=CollaborationDetail,
        summary="Collaboration state and history of one label",
        description=COLLABORATION_DESCRIPTION,
        **common,
    )
    def label_collaboration(resource_id: str, label: LabelName, review=Depends(source)) -> dict:
        if label_review.resource_report(resource_id, review[0]) is None:
            raise HTTPException(status_code=404, detail="Unknown resource")
        return detail(review, resource_id, label)

    @app.put(
        item + "/assignment",
        response_model=CollaborationDetail,
        summary="Assign or reassign a label",
        description="Assigns the label to `assignee`; reassigning to the current assignee "
        "records nothing. " + COLLABORATION_DESCRIPTION,
        **common,
    )
    def put_assignment(
        resource_id: str, label: LabelName, req: AssignmentRequest, review=Depends(source)
    ) -> dict:
        inventory, store = review
        _collaboration_write(
            store,
            lambda: store.assign(
                resource_id, label, req.assignee, inventory, reviewer=req.reviewer
            ),
        )
        return detail(review, resource_id, label)

    @app.delete(
        item + "/assignment",
        response_model=CollaborationDetail,
        summary="Unassign a label",
        description="Records an `unassign` event. 404, recording nothing, when unassigned. "
        "The optional JSON body supplies `reviewer`.",
        **common,
    )
    def delete_assignment(
        resource_id: str,
        label: LabelName,
        req: ReviewerRequest | None = None,
        review=Depends(source),
    ) -> dict:
        store = review[1]
        reviewer = (req or ReviewerRequest()).reviewer
        event = _collaboration_write(
            store, lambda: store.unassign(resource_id, label, reviewer=reviewer)
        )
        if event is None:
            raise HTTPException(status_code=404, detail="Not assigned")
        return detail(review, resource_id, label)

    @app.post(
        item + "/notes",
        response_model=CollaborationDetail,
        summary="Add a review note",
        description="Appends a short free-text note. " + COLLABORATION_DESCRIPTION,
        **common,
    )
    def post_note(
        resource_id: str, label: LabelName, req: NoteRequest, review=Depends(source)
    ) -> dict:
        inventory, store = review
        _collaboration_write(
            store,
            lambda: store.add_note(resource_id, label, req.text, inventory, reviewer=req.reviewer),
        )
        return detail(review, resource_id, label)

    @app.post(
        item + "/questions",
        response_model=CollaborationDetail,
        summary="Open an unresolved question",
        description="The new question's `id` is in `questions`. It stays unresolved (open, then "
        "answered) until resolved. " + COLLABORATION_DESCRIPTION,
        **common,
    )
    def post_question(
        resource_id: str, label: LabelName, req: NoteRequest, review=Depends(source)
    ) -> dict:
        inventory, store = review
        _collaboration_write(
            store,
            lambda: store.ask_question(
                resource_id, label, req.text, inventory, reviewer=req.reviewer
            ),
        )
        return detail(review, resource_id, label)

    @app.post(
        item + "/questions/{question_id}/answer",
        response_model=CollaborationDetail,
        summary="Answer a question",
        description="404 for a question on another label; 422 once resolved.",
        **common,
    )
    def post_answer(
        resource_id: str,
        label: LabelName,
        question_id: int,
        req: NoteRequest,
        review=Depends(source),
    ) -> dict:
        store = review[1]
        _collaboration_write(
            store,
            lambda: store.answer_question(
                question_id, req.text, reviewer=req.reviewer, resource_id=resource_id, label=label
            ),
        )
        return detail(review, resource_id, label)

    @app.post(
        item + "/questions/{question_id}/resolve",
        response_model=CollaborationDetail,
        summary="Resolve a question",
        description="Resolving is not an approval; the label still needs a decision.",
        **common,
    )
    def post_resolve(
        resource_id: str,
        label: LabelName,
        question_id: int,
        req: ResolveRequest | None = None,
        review=Depends(source),
    ) -> dict:
        store = review[1]
        req = req or ResolveRequest()
        _collaboration_write(
            store,
            lambda: store.resolve_question(
                question_id, req.text, reviewer=req.reviewer, resource_id=resource_id, label=label
            ),
        )
        return detail(review, resource_id, label)


_register_collaboration(
    "", "/labels/collaboration", synthetic_review, [Depends(require_api_key)], "labels"
)
_register_collaboration(
    "/reviews/{review_id}",
    "/reviews/{review_id}/collaboration",
    saved_review,
    REVIEW_DEPENDENCIES,
    "reviews",
)
