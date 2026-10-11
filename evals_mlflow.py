"""
Live evals recorded in MLflow: one run per provider/model.

Same cases and fact checks as evals.py, plus a trace-based check that the
agent actually called the tools each question needs (a model that guesses
can pass the fact check by luck, but not this one).

    uv sync --group mlops
    uv run evals_mlflow.py --provider deepseek
    uv run evals_mlflow.py --provider ollama --model qwen2.5:7b
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # compare runs

Live runs call the provider's API (paid for hosted providers).
"""

from __future__ import annotations

import argparse
import os
import subprocess

os.environ.setdefault("OBS_AGENT_MLFLOW", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
from mlflow.entities import Feedback, SpanType  # noqa: E402
from mlflow.genai import scorer  # noqa: E402

import observability  # noqa: E402
from agent import default_model, default_provider, run  # noqa: E402
from evals import EVALS, _check_eval  # noqa: E402


def accepted_tool_sets(case: dict) -> list[list[str]]:
    """Tool sets that each count as answering properly.

    Default: the tools the case's dry run calls. A case can list alternatives
    in `accepted_tools` when several tool routes are equally valid.
    """
    if "accepted_tools" in case:
        return case["accepted_tools"]
    spec = case["dry_run_tool"]
    pairs = spec if isinstance(spec, list) else [spec]
    return [[name for name, _ in pairs]]


def dataset() -> list[dict]:
    return [
        {
            "inputs": {"question": c["question"]},
            "expectations": {
                "case_id": c["id"],
                "must_contain": c["must_contain"],
                "accepted_tools": accepted_tool_sets(c),
            },
        }
        for c in EVALS
    ]


@scorer
def facts_present(outputs, expectations) -> Feedback:
    ok, missing = _check_eval(str(outputs or ""), expectations["must_contain"])
    return Feedback(value=ok, rationale="all facts present" if ok else f"missing: {missing}")


@scorer
def tools_called(trace, expectations) -> Feedback:
    called = [sp.name for sp in trace.search_spans(span_type=SpanType.TOOL)]
    options = expectations["accepted_tools"]
    ok = any(set(option) <= set(called) for option in options)
    if ok:
        return Feedback(value=True, rationale=f"called {called}")
    needed = " or ".join("+".join(o) for o in options)
    return Feedback(value=False, rationale=f"called {called}; needed {needed}")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_mlflow_evals(provider: str, model: str):
    observability.setup()

    def predict_fn(question: str) -> str:
        answer, _ = run(question, provider=provider, model=model)
        return answer

    with mlflow.start_run(run_name=f"{provider}/{model}"):
        mlflow.log_params(
            {"provider": provider, "model": model, "cases": len(EVALS), "git_sha": _git_sha()}
        )
        return mlflow.genai.evaluate(
            data=dataset(),
            predict_fn=predict_fn,
            scorers=[facts_present, tools_called],
        )


def summarise(results) -> str:
    """Per-case PASS/FAIL from the run's traces, then the aggregate scores."""
    exp_id = mlflow.get_run(results.run_id).info.experiment_id
    traces = mlflow.search_traces(locations=[exp_id], run_id=results.run_id, return_type="list")
    rows = []
    for t in traces:
        verdict = {a.name: a.value for a in t.info.assessments if isinstance(a, Feedback)}
        case = next(a.value for a in t.info.assessments if a.name == "case_id")
        notes = [
            label
            for key, label in (("facts_present", "facts missing"), ("tools_called", "tool skipped"))
            if not verdict.get(key)
        ]
        rows.append(
            f"[{'FAIL' if notes else 'PASS'}] {case}" + (f"  ({', '.join(notes)})" if notes else "")
        )
    m = results.metrics
    rows.sort(key=lambda r: [c["id"] for c in EVALS].index(r.split("] ")[1].split()[0]))
    rows.append(
        f"\nfacts present: {m.get('facts_present/mean', 0):.0%}   "
        f"tools called: {m.get('tools_called/mean', 0):.0%}   run: {results.run_id}"
    )
    return "\n".join(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live evals recorded in MLflow")
    parser.add_argument("--provider", help="Default: $OBS_AGENT_PROVIDER, else ollama.")
    parser.add_argument("--model", help="Default: the provider's default model.")
    args = parser.parse_args()
    provider = args.provider or default_provider()
    model = args.model or default_model(provider)
    print(summarise(run_mlflow_evals(provider, model)))
