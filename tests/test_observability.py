"""
Tests for optional MLflow tracing (observability.py) and the MLflow eval
runner (evals_mlflow.py). The "off" tests run everywhere; the rest need the
mlops dependency group and are skipped without it (CI's mlops job runs them).
No live LLM calls: Provider.create is patched.
"""

import contextlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import observability


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("OBS_AGENT_MLFLOW", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    observability.reset()
    yield
    observability.reset()


class TestOff:
    def test_not_requested_by_default(self):
        assert not observability.requested()
        assert observability.setup() is False
        assert not observability.active()

    def test_span_is_noop(self):
        with observability.span("x", "TOOL", inputs={"a": 1}) as s:
            assert s is None
        observability.set_outputs(None, "ignored")
        assert observability.last_trace_id() is None

    def test_missing_mlflow_warns_and_continues(self, monkeypatch, caplog):
        import builtins

        monkeypatch.setenv("OBS_AGENT_MLFLOW", "1")
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mlflow":
                raise ImportError("no mlflow")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert observability.setup() is False
        assert "uv sync --group mlops" in caplog.text


# ── Degraded tracing (fake mlflow module; runs with or without mlops) ──


def _answer(text="sub-a is top"):
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(
        stop_reason="end_turn", content=[block], model="m", usage=None, stop_details=None
    )


def _ask(question="Which is top?", **create):
    import agent

    create = create or {"return_value": _answer()}
    with (
        patch("agent.Provider.__init__", return_value=None),
        patch("agent.Provider.create", **create),
    ):
        return agent.run(question, provider="anthropic", model="m")


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(observability, "time", c)
    return c


@pytest.fixture
def fake_mlflow(monkeypatch):
    """A stand-in mlflow whose tracking store is down until `.recover()`."""
    fake = MagicMock(name="mlflow")
    fake.set_experiment.side_effect = ConnectionError(
        "tracking store unavailable at https://alice:s3cret@mlflow.example.com/api?token=t0k"
    )
    fake.openai.autolog.return_value = None
    fake.anthropic.autolog.return_value = None
    spans = []

    @contextlib.contextmanager
    def start_span(name, span_type):
        s = MagicMock(name=f"span:{name}")
        spans.append((name, s))
        yield s

    fake.start_span.side_effect = start_span
    fake.spans = spans
    fake.recover = lambda: setattr(fake.set_experiment, "side_effect", None)
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI", "https://alice:s3cret@mlflow.example.com/api?token=t0k"
    )
    return fake


class TestDegraded:
    def test_unavailable_store_does_not_fail_request(self, fake_mlflow, clock, caplog):
        answer, _ = _ask()
        assert answer == "sub-a is top"
        assert not observability.active()
        st = observability.status()
        assert st["attempts"] == 1 and st["disabled_reason"] is None
        assert "ConnectionError" in caplog.text
        assert "https://mlflow.example.com" in caplog.text
        assert "retrying in 60s" in caplog.text

    def test_warning_omits_credentials_and_request_content(self, fake_mlflow, clock, caplog):
        _ask("secret question about acme-prod")
        for leaked in ("s3cret", "alice", "t0k", "/api", "acme-prod"):
            assert leaked not in caplog.text

    def test_subsequent_requests_retry_after_interval_then_recover(
        self, fake_mlflow, clock, caplog
    ):
        _ask()
        _ask()  # within the retry interval: no new attempt, no new warning
        assert fake_mlflow.set_experiment.call_count == 1
        assert caplog.text.count("MLflow tracing degraded") == 1

        clock.now += observability.RETRY_INTERVAL_S
        fake_mlflow.recover()
        answer, _ = _ask()
        assert answer == "sub-a is top"
        assert fake_mlflow.set_experiment.call_count == 2
        assert observability.active()
        assert observability.status()["attempts"] == 2
        assert "agent" in [name for name, _ in fake_mlflow.spans]

    def test_bounded_retries_then_documented_disabled_state(self, fake_mlflow, clock, caplog):
        for _ in range(observability.MAX_SETUP_ATTEMPTS + 2):
            assert _ask()[0] == "sub-a is top"
            clock.now += observability.RETRY_INTERVAL_S
        assert fake_mlflow.set_experiment.call_count == observability.MAX_SETUP_ATTEMPTS
        st = observability.status()
        assert st["active"] is False
        assert st["disabled_reason"] == "tracking store unavailable (ConnectionError)"
        assert "disabled for this process" in caplog.text
        fake_mlflow.recover()
        assert observability.setup() is False  # stays disabled until restart

    def test_tracing_disabled_never_touches_mlflow(self, fake_mlflow, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI")
        assert _ask()[0] == "sub-a is top"
        assert observability.status()["disabled_reason"] == "not requested"
        fake_mlflow.set_experiment.assert_not_called()
        fake_mlflow.start_span.assert_not_called()

    def test_missing_mlflow_request_completes(self, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, "mlflow", None)  # import raises ImportError
        monkeypatch.setenv("OBS_AGENT_MLFLOW", "1")
        assert _ask()[0] == "sub-a is top"
        assert observability.status()["disabled_reason"] == "mlflow not installed"
        assert "uv sync --group mlops" in caplog.text

    def test_span_recording_failure_does_not_fail_request(self, fake_mlflow, clock, caplog):
        fake_mlflow.recover()
        fake_mlflow.start_span.side_effect = ConnectionError("store went away")
        assert observability.setup() is True
        assert _ask()[0] == "sub-a is top"
        assert caplog.text.count("start_span failed") == 1


class _ClosingFails:
    """A span context manager whose __exit__ raises, as a failing tracking store would."""

    def __init__(self, exc):
        self.exc = exc
        self.exited_with = None

    def __enter__(self):
        return MagicMock(name="span")

    def __exit__(self, exc_type, exc, tb):
        self.exited_with = exc_type
        raise self.exc


class TestSpanClose:
    def _activate(self, fake_mlflow, cm):
        fake_mlflow.recover()
        fake_mlflow.start_span.side_effect = lambda **kw: cm
        assert observability.setup() is True

    def test_close_failure_after_success_does_not_propagate(self, fake_mlflow, clock, caplog):
        cm = _ClosingFails(ConnectionError("store went away"))
        self._activate(fake_mlflow, cm)
        with observability.span("tool", "TOOL") as s:
            result = "done"
        assert s is not None and result == "done"
        assert cm.exited_with is None
        assert "span close failed (ConnectionError)" in caplog.text
        assert _ask()[0] == "sub-a is top"  # a whole request survives it too

    def test_body_exception_propagates_unchanged_when_close_also_fails(
        self, fake_mlflow, clock, caplog
    ):
        cm = _ClosingFails(ConnectionError("store went away"))
        self._activate(fake_mlflow, cm)
        boom = RuntimeError("caller failure")
        with pytest.raises(RuntimeError) as info, observability.span("tool", "TOOL"):
            raise boom
        assert info.value is boom
        assert cm.exited_with is RuntimeError  # MLflow still saw the caller's error
        assert "span close failed" in caplog.text

    def test_body_exception_propagates_when_span_reraises_it(self, fake_mlflow, clock):
        boom = RuntimeError("caller failure")
        cm = _ClosingFails(boom)
        self._activate(fake_mlflow, cm)
        with pytest.raises(RuntimeError) as info, observability.span("tool", "TOOL"):
            raise boom
        assert info.value is boom


class TestAgentErrorsStillRaised:
    def test_with_failed_tracing_init(self, fake_mlflow, clock):
        with pytest.raises(RuntimeError, match="provider exploded"):
            _ask(side_effect=RuntimeError("provider exploded"))
        assert fake_mlflow.set_experiment.call_count == 1

    def test_with_active_tracing(self, fake_mlflow, clock):
        fake_mlflow.recover()
        with pytest.raises(RuntimeError, match="provider exploded"):
            _ask(side_effect=RuntimeError("provider exploded"))
        assert observability.active()

    def test_with_tracing_off(self):
        with pytest.raises(RuntimeError, match="provider exploded"):
            _ask(side_effect=RuntimeError("provider exploded"))


# ── With MLflow installed ──────────────────────────────────────────────

try:
    import mlflow
except ImportError:  # the CI test jobs; only the mlops job installs it
    mlflow = None

needs_mlflow = pytest.mark.skipif(mlflow is None, reason="needs the mlops dependency group")


def _tool_then_answer():
    tool_block = SimpleNamespace(
        type="tool_use", id="t1", name="get_label_coverage", input={"limit": 1}
    )
    first = SimpleNamespace(
        stop_reason="tool_use", content=[tool_block], model="m", usage=None, stop_details=None
    )
    text = SimpleNamespace(type="text", text="170 untagged")
    second = SimpleNamespace(
        stop_reason="end_turn", content=[text], model="m", usage=None, stop_details=None
    )
    return [first, second]


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("OBS_AGENT_MLFLOW_EXPERIMENT", "test-obs-agent")
    assert observability.setup() is True
    yield


@needs_mlflow
class TestTracing:
    def test_agent_run_produces_nested_trace(self, tracking):
        import agent

        with (
            patch("agent.Provider.__init__", return_value=None),
            patch("agent.Provider.create", side_effect=_tool_then_answer()),
        ):
            answer, _ = agent.run("how many untagged?", provider="anthropic", model="m")
        assert answer == "170 untagged"

        trace_id = observability.last_trace_id()
        mlflow.flush_trace_async_logging()
        trace = mlflow.get_trace(trace_id)
        spans = {s.name: s for s in trace.data.spans}
        assert {"agent", "llm: anthropic/m", "get_label_coverage"} <= set(spans)
        root = spans["agent"]
        assert root.parent_id is None
        assert spans["get_label_coverage"].parent_id == root.span_id
        assert spans["get_label_coverage"].span_type == "TOOL"
        assert root.outputs["answer"] == "170 untagged"
        llm_outputs = [s.outputs for s in trace.data.spans if s.name.startswith("llm:")]
        assert llm_outputs[0]["tool_calls"] == ["get_label_coverage"]


@needs_mlflow
class TestEvalRunner:
    def test_accepted_tool_sets(self):
        import evals_mlflow

        cases = {c["id"]: c for c in evals_mlflow.EVALS}
        assert evals_mlflow.accepted_tool_sets(cases["label-count-and-team"]) == [
            ["get_label_coverage", "suggest_labels"]
        ]
        assert evals_mlflow.accepted_tool_sets(cases["gcp-top-project"]) == [
            ["get_gcp_top_projects"]
        ]
        assert ["get_azure_top_overages", "get_gcp_top_projects"] in (
            evals_mlflow.accepted_tool_sets(cases["multi-cloud-summary"])
        )
        assert len(evals_mlflow.dataset()) == len(evals_mlflow.EVALS)

    def test_scorers(self):
        import evals_mlflow

        exp = {"must_contain": ["170", "data-platform"], "accepted_tools": [["suggest_labels"]]}
        assert evals_mlflow.facts_present(outputs="170 … data-platform", expectations=exp).value
        missing = evals_mlflow.facts_present(outputs="320 … web", expectations=exp)
        assert missing.value is False and "170" in missing.rationale

        fake_trace = SimpleNamespace(
            search_spans=lambda span_type=None: [SimpleNamespace(name="get_label_coverage")]
        )
        skipped = evals_mlflow.tools_called(trace=fake_trace, expectations=exp)
        assert skipped.value is False and "suggest_labels" in skipped.rationale

        either = {"accepted_tools": [["compare_cross_cloud"], ["a", "b"]]}
        two = SimpleNamespace(
            search_spans=lambda span_type=None: [
                SimpleNamespace(name="a"),
                SimpleNamespace(name="b"),
            ]
        )
        assert evals_mlflow.tools_called(trace=two, expectations=either).value
