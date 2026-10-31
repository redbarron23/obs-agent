"""
Optional MLflow tracing for the agent.

Off unless OBS_AGENT_MLFLOW=1 or MLFLOW_TRACKING_URI is set, and MLflow is
installed (`uv sync --group mlops`). When off, every helper here is a no-op,
so the CLI, API, UI, CI and Docker image never import MLflow. Tracing is
best-effort: if the tracking store is unreachable, setup() warns and the
request runs untraced (bounded retries, then disabled; see MAX_SETUP_ATTEMPTS).

One question becomes one trace:

    agent (AGENT)
    ├── llm: <provider>/<model> (CHAT_MODEL)   one per model call
    │   └── provider SDK span                  when MLflow autolog patches it
    ├── <tool name> (TOOL)                     one per tool call
    └── ...

MLflow's Anthropic autolog patches only `messages.create`, not the beta
endpoint or `messages.stream` that Claude Opus 5.5 / streaming use, so the
provider-neutral `llm:` span is what guarantees every model call is visible.

Run `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db` to browse.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from urllib.parse import urlsplit

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "obs-agent"

# Bounded recovery when the tracking store fails during configuration: each
# failure leaves tracing off (the agent request still runs), a later setup()
# retries after RETRY_INTERVAL_S, and after MAX_SETUP_ATTEMPTS failures
# tracing stays disabled until the process restarts.
MAX_SETUP_ATTEMPTS = 3
RETRY_INTERVAL_S = 60.0

_log = logging.getLogger(__name__)
_lock = threading.Lock()
_state: dict = {}


def _initial_state() -> dict:
    return {
        "checked": False,  # True once configuration is final (active or disabled)
        "active": False,
        "attempts": 0,
        "retry_at": 0.0,
        "disabled_reason": None,
        "span_warned": False,
    }


_state.update(_initial_state())


def requested() -> bool:
    """Whether the environment asks for MLflow tracing."""
    return os.environ.get("OBS_AGENT_MLFLOW") == "1" or bool(os.environ.get("MLFLOW_TRACKING_URI"))


def _redacted_uri(uri: str) -> str:
    """Tracking destination safe to log: no credentials, path, query or fragment."""
    parts = urlsplit(uri)
    if not parts.scheme:
        return "<custom>" if "@" in uri else uri
    if parts.netloc:
        host = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""
        return f"{parts.scheme}://{host}{port}"
    return f"{parts.scheme}://{parts.path}" if parts.scheme == "sqlite" else f"{parts.scheme}:"


def _configure(mlflow) -> None:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
    mlflow.set_experiment(os.environ.get("OBS_AGENT_MLFLOW_EXPERIMENT", DEFAULT_EXPERIMENT))
    # Nested SDK detail where MLflow can patch the client (see module docstring).
    for flavor in ("openai", "anthropic"):
        try:
            getattr(mlflow, flavor).autolog(silent=True)
        except Exception as exc:  # optional extra; the agent's own spans still work
            _log.debug("mlflow.%s.autolog unavailable: %s", flavor, exc)


def _disable(reason: str) -> bool:
    _state.update(checked=True, active=False, disabled_reason=reason)
    return False


def setup() -> bool:
    """Configure MLflow for this process; returns whether tracing is active.

    Never raises: tracing is optional, so a missing package or an unreachable
    tracking store degrades to untraced requests with a warning. A failed
    configuration is not recorded as done; see MAX_SETUP_ATTEMPTS.
    """
    with _lock:
        if _state["checked"]:
            return _state["active"]
        if not requested():
            return _disable("not requested")
        if time.monotonic() < _state["retry_at"]:
            return False
        try:
            import mlflow
        except ImportError:
            _log.warning(
                "MLflow tracing requested but mlflow is not installed; "
                "run `uv sync --group mlops`. Continuing without tracing."
            )
            return _disable("mlflow not installed")
        _state["attempts"] += 1
        try:
            _configure(mlflow)
        except Exception as exc:
            # Log the error type and a redacted destination only: the message
            # can embed credentials from the tracking URI.
            dest = _redacted_uri(os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI)
            attempt = _state["attempts"]
            if attempt >= MAX_SETUP_ATTEMPTS:
                _log.warning(
                    "MLflow tracing disabled for this process after %d failed attempts "
                    "(last: %s configuring tracking store %s). Requests continue untraced; "
                    "check MLFLOW_TRACKING_URI and the tracking server, then restart.",
                    attempt,
                    type(exc).__name__,
                    dest,
                )
                return _disable(f"tracking store unavailable ({type(exc).__name__})")
            _state["retry_at"] = time.monotonic() + RETRY_INTERVAL_S
            _log.warning(
                "MLflow tracing degraded: %s configuring tracking store %s (attempt %d/%d). "
                "This request continues untraced; retrying in %.0fs. "
                "Check MLFLOW_TRACKING_URI and that the tracking server is reachable.",
                type(exc).__name__,
                dest,
                attempt,
                MAX_SETUP_ATTEMPTS,
                RETRY_INTERVAL_S,
            )
            return False
        _state.update(checked=True, active=True, disabled_reason=None)
        return True


def active() -> bool:
    return _state["active"]


def status() -> dict:
    """Tracing state for diagnostics: active, attempts, and why it is off."""
    return {k: _state[k] for k in ("active", "attempts", "disabled_reason")}


def reset() -> None:
    """Forget the configuration (tests)."""
    with _lock:
        _state.update(_initial_state())


def _telemetry_failed(what: str, exc: Exception) -> None:
    """Report a span-recording failure once per process; never raise."""
    if not _state["span_warned"]:
        _state["span_warned"] = True
        _log.warning(
            "MLflow %s failed (%s); continuing without that trace data.",
            what,
            type(exc).__name__,
        )
    else:
        _log.debug("MLflow %s failed (%s)", what, type(exc).__name__)


@contextlib.contextmanager
def span(name: str, span_type: str, inputs: dict | None = None, attributes: dict | None = None):
    """An MLflow span when tracing is active, else a no-op yielding None.

    Callers set outputs with `set_outputs(s, ...)`, which tolerates None.
    """
    if not _state["active"]:
        yield None
        return
    import mlflow

    with contextlib.ExitStack() as stack:
        try:
            s = stack.enter_context(mlflow.start_span(name=name, span_type=span_type))
        except Exception as exc:
            _telemetry_failed("start_span", exc)
            s = None
        if s is not None:
            try:
                if inputs is not None:
                    s.set_inputs(inputs)
                if attributes:
                    s.set_attributes(attributes)
            except Exception as exc:
                _telemetry_failed("span inputs", exc)
        # Outside any telemetry try/except: exceptions from the caller's body
        # propagate unchanged (MLflow's span records them and re-raises).
        yield s


def set_outputs(s, outputs) -> None:
    if s is not None:
        try:
            s.set_outputs(outputs)
        except Exception as exc:
            _telemetry_failed("span outputs", exc)


def last_trace_id() -> str | None:
    if not _state["active"]:
        return None
    import mlflow

    return mlflow.get_last_active_trace_id()
