"""
Optional MLflow tracing for the agent.

Off unless OBS_AGENT_MLFLOW=1 or MLFLOW_TRACKING_URI is set, and MLflow is
installed (`uv sync --group mlops`). When off, every helper here is a no-op,
so the CLI, API, UI, CI and Docker image never import MLflow.

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

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "obs-agent"

_log = logging.getLogger(__name__)
_lock = threading.Lock()
_state = {"checked": False, "active": False}


def requested() -> bool:
    """Whether the environment asks for MLflow tracing."""
    return os.environ.get("OBS_AGENT_MLFLOW") == "1" or bool(os.environ.get("MLFLOW_TRACKING_URI"))


def setup() -> bool:
    """Configure MLflow once per process; returns whether tracing is active."""
    with _lock:
        if _state["checked"]:
            return _state["active"]
        _state["checked"] = True
        if not requested():
            return False
        try:
            import mlflow
        except ImportError:
            _log.warning(
                "MLflow tracing requested but mlflow is not installed; "
                "run `uv sync --group mlops`. Continuing without tracing."
            )
            return False
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
        mlflow.set_experiment(os.environ.get("OBS_AGENT_MLFLOW_EXPERIMENT", DEFAULT_EXPERIMENT))
        # Nested SDK detail where MLflow can patch the client (see module docstring).
        for flavor in ("openai", "anthropic"):
            try:
                getattr(mlflow, flavor).autolog(silent=True)
            except Exception as exc:  # optional extra; the agent's own spans still work
                _log.debug("mlflow.%s.autolog unavailable: %s", flavor, exc)
        _state["active"] = True
        return True


def active() -> bool:
    return _state["active"]


def reset() -> None:
    """Forget the configuration (tests)."""
    with _lock:
        _state.update(checked=False, active=False)


@contextlib.contextmanager
def span(name: str, span_type: str, inputs: dict | None = None, attributes: dict | None = None):
    """An MLflow span when tracing is active, else a no-op yielding None.

    Callers set outputs with `set_outputs(s, ...)`, which tolerates None.
    """
    if not _state["active"]:
        yield None
        return
    import mlflow

    with mlflow.start_span(name=name, span_type=span_type) as s:
        if inputs is not None:
            s.set_inputs(inputs)
        if attributes:
            s.set_attributes(attributes)
        yield s


def set_outputs(s, outputs) -> None:
    if s is not None:
        s.set_outputs(outputs)


def last_trace_id() -> str | None:
    if not _state["active"]:
        return None
    import mlflow

    return mlflow.get_last_active_trace_id()
