"""
Operational metrics for the inventory-review workflow (stdlib only).

    import → inference → decision write → export

Each step runs inside `operation(name)`, which counts the outcome, times it,
and writes one structured JSON log line carrying an operation name and a
correlation ID. The API returns that ID in the `X-Request-ID` header and the
UI shows it next to an error, so an operator can find the server-side cause.

Metric labels are bounded: `operation` (OPERATIONS), `surface` (SURFACES) and
`outcome` (OUTCOMES). Nothing derived from the inventory, the reviewer, or an
exception message is ever a label or part of a routine log line: failures log
the exception *type* only, and the full traceback only at DEBUG.

Telemetry is best-effort. A failure while recording a measurement is
swallowed (see `_telemetry_failed`), so it can never fail an import, review,
or export. Counters are in-process; `snapshot()` / `prometheus()` read them
and `backlog()` measures pending review work. See docs/operations.md.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from datetime import UTC, datetime

OPERATIONS = ("import", "inference", "decision_write", "export")
SURFACES = ("api", "ui", "other")
# success: completed. invalid: rejected user input (bad CSV, bad decision
# value); expected, not an alert. error: anything else; investigate.
OUTCOMES = ("success", "invalid", "error")
DURATION_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)

_log = logging.getLogger("obs_agent.workflow")
if level := os.environ.get("OBS_AGENT_LOG_LEVEL"):
    logging.basicConfig(
        level=level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    _log.setLevel(level.upper())

_lock = threading.Lock()
_counts: dict[tuple[str, str, str], int] = {}
_durations: dict[tuple[str, str], dict] = {}
_telemetry_errors = {"count": 0}

_correlation: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "obs_agent_correlation_id", default=None
)
_surface: contextvars.ContextVar[str] = contextvars.ContextVar("obs_agent_surface", default="other")
_CORRELATION_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")


def new_correlation_id() -> str:
    return uuid.uuid4().hex[:16]


def clean_correlation_id(value: str | None) -> str:
    """A caller-supplied request ID if it is safe to echo and log, else a new one."""
    if value and _CORRELATION_RE.fullmatch(value):
        return value
    return new_correlation_id()


def bind(correlation_id: str | None = None, surface: str | None = None):
    """Set the request's correlation ID and surface; returns tokens for `unbind`."""
    tokens = []
    if correlation_id is not None:
        tokens.append(_correlation.set(correlation_id))
    if surface is not None:
        tokens.append(_surface.set(surface if surface in SURFACES else "other"))
    return tokens


def unbind(tokens) -> None:
    for token in reversed(tokens):
        token.var.reset(token)


def current_correlation_id() -> str | None:
    return _correlation.get()


def reset() -> None:
    """Forget all measurements (tests)."""
    with _lock:
        _counts.clear()
        _durations.clear()
        _telemetry_errors["count"] = 0


def _telemetry_failed(exc: Exception) -> None:
    """A measurement could not be recorded: note it and carry on; never raise."""
    try:
        _telemetry_errors["count"] += 1
        _log.debug("workflow telemetry failed (%s)", type(exc).__name__)
    except Exception:
        pass


def _record(operation: str, surface: str, outcome: str, seconds: float) -> None:
    with _lock:
        key = (operation, surface, outcome)
        _counts[key] = _counts.get(key, 0) + 1
        d = _durations.setdefault(
            (operation, surface),
            {"count": 0, "sum": 0.0, "max": 0.0, "buckets": [0] * len(DURATION_BUCKETS)},
        )
        d["count"] += 1
        d["sum"] += seconds
        d["max"] = max(d["max"], seconds)
        for i, bound in enumerate(DURATION_BUCKETS):
            if seconds <= bound:
                d["buckets"][i] += 1


def _emit(event: dict, level: int, exc: BaseException | None) -> None:
    _log.log(level, json.dumps(event, sort_keys=True), extra={"workflow": event})
    if exc is not None and _log.isEnabledFor(logging.DEBUG):
        # Tracebacks can quote inventory values or reviewer input; opt-in only.
        _log.debug("traceback for correlation_id=%s", event["correlation_id"], exc_info=exc)


def log_unhandled(correlation_id: str | None, route: str, method: str, exc: Exception) -> None:
    """Structured log line for an unexpected request failure; never raises."""
    try:
        event = {
            "event": "request_failed",
            "route": route,
            "method": method,
            "correlation_id": correlation_id,
            "error_type": type(exc).__name__,
        }
        _emit(event, logging.ERROR, exc)
    except Exception as telemetry_exc:
        _telemetry_failed(telemetry_exc)


class operation:
    """Measure one workflow step: `with operation("export") as op: ...`.

    The body's exceptions propagate unchanged. Exceptions of the `invalid`
    types count as outcome "invalid" (rejected input); any other exception
    is an "error". `op.correlation_id` is available before, during and after
    the block, for showing to the user alongside an error.
    """

    def __init__(
        self,
        name: str,
        *,
        surface: str | None = None,
        invalid: tuple[type[BaseException], ...] = (),
        correlation_id: str | None = None,
    ):
        if name not in OPERATIONS:
            raise ValueError(f"Unknown workflow operation '{name}'")
        self.name = name
        self.surface = surface if surface in SURFACES else _surface.get()
        self.invalid = invalid
        self.correlation_id = correlation_id or _correlation.get() or new_correlation_id()
        self.outcome: str | None = None
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            seconds = time.perf_counter() - self._start
            if exc is None:
                self.outcome, level = "success", logging.INFO
            elif isinstance(exc, self.invalid):
                self.outcome, level = "invalid", logging.WARNING
            else:
                self.outcome, level = "error", logging.ERROR
            _record(self.name, self.surface, self.outcome, seconds)
            event = {
                "event": "workflow_operation",
                "operation": self.name,
                "surface": self.surface,
                "outcome": self.outcome,
                "correlation_id": self.correlation_id,
                "duration_ms": round(seconds * 1000, 1),
            }
            if exc is not None:
                event["error_type"] = type(exc).__name__
            _emit(event, level, exc)
        except Exception as telemetry_exc:
            _telemetry_failed(telemetry_exc)
        return False  # never swallow the caller's exception


def snapshot() -> dict:
    """Counters and duration summaries recorded by this process."""
    with _lock:
        operations = [
            {"operation": o, "surface": s, "outcome": r, "count": n}
            for (o, s, r), n in sorted(_counts.items())
        ]
        durations = [
            {
                "operation": o,
                "surface": s,
                "count": d["count"],
                "sum_seconds": round(d["sum"], 6),
                "max_seconds": round(d["max"], 6),
                "buckets": dict(zip((str(b) for b in DURATION_BUCKETS), d["buckets"], strict=True)),
            }
            for (o, s), d in sorted(_durations.items())
        ]
        telemetry_errors = _telemetry_errors["count"]
    return {
        "operations": operations,
        "durations": durations,
        "telemetry_errors": telemetry_errors,
    }


# ── Review backlog ─────────────────────────────────────────────────────


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _workspace_created_at(store) -> datetime | None:
    try:
        with store._connect() as conn:
            if not conn.execute(
                "SELECT name FROM sqlite_master WHERE name = 'workspace'"
            ).fetchone():
                return None
            row = conn.execute("SELECT created_at FROM workspace WHERE id = 1").fetchone()
    except sqlite3.Error:
        return None
    return _parse(row[0]) if row else None


def backlog(store, inventory=None, now: datetime | None = None) -> dict:
    """Pending review work for the snapshot under review (see docs/operations.md).

    Pending work is a (resource, label) pair that is untagged or suspect and
    has no decision bound to the current snapshot and inference version.
    Feedback actions are the append-only history (including superseded,
    undone, and earlier-snapshot actions) and are reported separately.
    """
    import label_review
    from review_context import review_context

    now = now or datetime.now(UTC)
    context = review_context(inventory)
    current = {
        (d["resource_id"], d["label"]) for d in label_review.current_decisions(store, inventory)
    }
    decisions = store.all()
    pending = pending_auto = pending_abstained = 0
    pending_resources = set()
    for r in label_review.reports(inventory):
        for label, lr in r["labels"].items():
            if lr["status"] in ("untagged", "suspect") and (r["resource_id"], label) not in current:
                pending += 1
                pending_auto += lr["action"] == "auto-apply"
                pending_abstained += lr["action"] == label_review.INSUFFICIENT_EVIDENCE
                pending_resources.add(r["resource_id"])
    feedback = store.feedback()
    snapshot_feedback = sorted(
        ts
        for ts in (
            _parse(f["decided_at"])
            for f in feedback
            if all(f.get(k) == v for k, v in context.items())
        )
        if ts is not None
    )
    started = _workspace_created_at(store) or (snapshot_feedback[0] if snapshot_feedback else None)

    def age(ts):
        return None if ts is None else max(0, round((now - ts).total_seconds()))

    return {
        "pending_items": pending,
        "pending_resources": len(pending_resources),
        "pending_auto_apply_ready": pending_auto,
        "pending_insufficient_evidence": pending_abstained,
        "reviewed_items": len(current),
        "stale_decisions": len(decisions) - len(current),
        "feedback_actions": len(feedback),
        "review_started_at": started.isoformat(timespec="seconds") if started else None,
        "oldest_pending_age_seconds": age(started) if pending else 0,
        "last_action_age_seconds": age(snapshot_feedback[-1] if snapshot_feedback else None),
    }


# ── Prometheus text exposition ─────────────────────────────────────────


def prometheus(backlog_values: dict | None = None) -> str:
    """The counters (and optional backlog gauges) in Prometheus text format."""
    snap = snapshot()
    lines = [
        "# HELP obs_agent_workflow_operations_total Workflow operations by outcome.",
        "# TYPE obs_agent_workflow_operations_total counter",
    ]
    for row in snap["operations"]:
        lines.append(
            "obs_agent_workflow_operations_total"
            f'{{operation="{row["operation"]}",surface="{row["surface"]}",'
            f'outcome="{row["outcome"]}"}} {row["count"]}'
        )
    name = "obs_agent_workflow_operation_duration_seconds"
    lines += [
        f"# HELP {name} Workflow operation duration.",
        f"# TYPE {name} histogram",
    ]
    for row in snap["durations"]:
        labels = f'operation="{row["operation"]}",surface="{row["surface"]}"'
        for bound, n in row["buckets"].items():
            lines.append(f'{name}_bucket{{{labels},le="{bound}"}} {n}')
        lines.append(f'{name}_bucket{{{labels},le="+Inf"}} {row["count"]}')
        lines.append(f"{name}_sum{{{labels}}} {row['sum_seconds']}")
        lines.append(f"{name}_count{{{labels}}} {row['count']}")
    lines += [
        "# HELP obs_agent_workflow_telemetry_errors_total Measurements that failed to record.",
        "# TYPE obs_agent_workflow_telemetry_errors_total counter",
        f"obs_agent_workflow_telemetry_errors_total {snap['telemetry_errors']}",
    ]
    for key, value in (backlog_values or {}).items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        gauge = f"obs_agent_review_backlog_{key}"
        lines += [f"# TYPE {gauge} gauge", f"{gauge} {value}"]
    return "\n".join(lines) + "\n"
