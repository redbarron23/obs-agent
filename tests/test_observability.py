"""
Tests for optional MLflow tracing (observability.py) and the MLflow eval
runner (evals_mlflow.py). The "off" tests run everywhere; the rest need the
mlops dependency group and are skipped without it (CI's mlops job runs them).
No live LLM calls: Provider.create is patched.
"""

from types import SimpleNamespace
from unittest.mock import patch

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
