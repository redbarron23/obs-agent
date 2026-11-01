"""
Eval harness for the obs-agent.

Tests a fixed set of questions against known-correct answers
(verified from the deterministic synthetic data in data.py).

Because LLM output is non-deterministic, we check for *presence of
correct facts* (must_contain) rather than exact-string matches.

Dry-run mode checks tool outputs directly — fully deterministic, no API key.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import policy_docs
from agent import run
from tools import TOOL_DISPATCH, search_observability_docs

# Ground-truth expected identifiers.  These are stable because data.py
# uses a fixed random seed.
EVALS = [
    {
        "id": "azure-top-overage",
        "question": "Which Azure subscription has the highest overage cost?",
        "must_contain": ["sub-a1b2c3d4"],
        "dry_run_tool": ("get_azure_top_overages", {"n": 1}),
    },
    {
        "id": "gcp-top-project",
        "question": "Which GCP project has the highest logging cost?",
        "must_contain": ["project-alpha"],
        "dry_run_tool": ("get_gcp_top_projects", {"n": 1}),
    },
    {
        "id": "gcp-top-3",
        "question": "Show me the top 3 most expensive GCP projects",
        "must_contain": [
            "project-alpha",
            "project-bravo",
        ],
        "dry_run_tool": ("get_gcp_top_projects", {"n": 3}),
    },
    {
        "id": "multi-cloud-summary",
        "question": ("Give me a summary of the top overage on Azure and the top cost on GCP"),
        "must_contain": [
            "sub-a1b2c3d4",
            "project-alpha",
        ],
        "dry_run_tool": ("compare_cross_cloud", {}),
        # Either tool route answers it; evals_mlflow.py's tool check accepts both.
        "accepted_tools": [
            ["compare_cross_cloud"],
            ["get_azure_top_overages", "get_gcp_top_projects"],
        ],
    },
    {
        "id": "azure-daily-trend",
        "question": ("What is the daily cost trend for subscription sub-a1b2c3d4?"),
        "must_contain": ["sub-a1b2c3d4"],
        "dry_run_tool": (
            "get_daily_trend",
            {"platform": "azure", "identifier": "sub-a1b2c3d4"},
        ),
    },
    {
        "id": "gcp-daily-trend",
        "question": ("Show me the daily cost trend for project project-bravo"),
        "must_contain": ["project-bravo"],
        "dry_run_tool": (
            "get_daily_trend",
            {"platform": "gcp", "identifier": "project-bravo"},
        ),
    },
    {
        "id": "find-spikes-azure",
        "question": ("Find any cost spikes above 200% across Azure"),
        "must_contain": ["sub-a1b2c3d4"],
        "dry_run_tool": ("find_spikes", {"threshold_pct": 200}),
    },
    {
        "id": "find-spikes-gcp",
        "question": ("Find any cost spikes above 200% across GCP"),
        "must_contain": ["project-bravo"],
        "dry_run_tool": ("find_spikes", {"threshold_pct": 200, "platform": "gcp"}),
    },
    # Label inference — ground truth comes from inventory.py's true_* columns.
    {
        "id": "label-coverage",
        "question": "How many subscriptions and projects are missing a team tag?",
        "must_contain": ["170"],
        "dry_run_tool": ("get_label_coverage", {}),
    },
    {
        "id": "suggest-team",
        "question": "svc-ingest has no team tag. Which team does it probably belong to?",
        "must_contain": ["data-platform"],
        "dry_run_tool": ("suggest_labels", {"resource_id": "svc-ingest"}),
    },
    {
        "id": "suspect-label",
        "question": "Is the team tag on api-login-d-ew1 correct?",
        "must_contain": ["identity"],
        "dry_run_tool": ("suggest_labels", {"resource_id": "api-login-d-ew1"}),
    },
    # Regression for issue #5: a small local model reported the inventory total
    # (320) as the untagged count and guessed svc-ingest's team without a tool call.
    {
        "id": "label-count-and-team",
        "question": (
            "How many projects are missing a team tag, and which team does "
            "svc-ingest probably belong to?"
        ),
        "must_contain": ["170", "data-platform"],
        "dry_run_tool": [
            ("get_label_coverage", {}),
            ("suggest_labels", {"resource_id": "svc-ingest"}),
        ],
    },
]


# Documentation-retrieval cases: dry run only, against a stub of the rag-observability
# retrieval API (fixtures/), never the real service. Kept apart from EVALS so the live
# evals and evals_mlflow.py are unchanged when OBS_AGENT_RAG_URL is unset. A case with
# `dry_run_tool` also runs that cost tool for real, so it checks the billing facts and
# the cited runbook sources together (issue #91); `docs_question` is what the model would
# send to search_observability_docs, and `must_not_contain` lists what must be absent.
FIXTURES = Path(__file__).with_name("fixtures")
RAG_FIXTURE = FIXTURES / "rag_retrieve_tagging_standard.json"
# Captured from rag-observability's api.py over its fictional corpus, which includes
# data/runbooks/logging-cost-spike.md (see docs/cost-runbook-walkthrough.md).
RUNBOOK_CORPUS = "corpus_sha256=931f94bb"
RAG_DRY_RUN_EVALS = [
    {
        "id": "policy-env-values-cited",
        "question": "Which environment values does our tagging standard allow?",
        "must_contain": [
            "reference/tagging-standard.md",
            "Allowed values",
            "sandbox",
            "corpus_sha256=0a50d302",
            "freshness=current",
            "untrusted",
        ],
        "stub": (200, RAG_FIXTURE.name),
    },
    {
        "id": "policy-docs-unavailable",
        "question": "What does our standard require for the owner tag?",
        "must_contain": ["status: index_unavailable", "documentation guidance is unavailable"],
        "stub": (503, {"error": {"code": "index_unavailable", "message": "no index"}}),
    },
    # Cost findings plus cited runbook steps (issue #91).
    {
        "id": "cost-runbook-supported",
        "question": "Which GCP project has the largest logging spike, and what does our "
        "runbook say to investigate?",
        "dry_run_tool": ("find_spikes", {"threshold_pct": 200, "platform": "gcp"}),
        "docs_question": "What does our runbook say to investigate for a GCP logging cost spike?",
        "must_contain": [
            # Billing facts from the tool and the stated dataset and period.
            "project-bravo on 2026-04-23: +405% ($990.00)",
            "synthetic demo billing data",
            "2026-05-03 to 2026-06-01 (30 days)",
            "2026-04-03 to 2026-06-01",
            # Cited runbook source and section, and the corpus identity.
            'source="runbooks/logging-cost-spike.md" section="Investigation: GCP Cloud Logging"',
            "bytes_ingested",
            RUNBOOK_CORPUS,
            "status: ok",
        ],
        "must_not_contain": ["may conflict"],
        "stub": (200, "rag_retrieve_logging_cost_runbook.json"),
    },
    {
        "id": "cost-runbook-no-match",
        "question": "Which GCP project has the largest logging spike, and what does our "
        "runbook say about its spend that day?",
        "dry_run_tool": ("find_spikes", {"threshold_pct": 200, "platform": "gcp"}),
        "docs_question": "How much did project-bravo spend on logging on 23 April?",
        "must_contain": [
            "project-bravo on 2026-04-23: +405% ($990.00)",
            "2026-05-03 to 2026-06-01 (30 days)",
            "status: no_match",
            "documentation guidance is unavailable",
            RUNBOOK_CORPUS,
        ],
        "must_not_contain": ["<untrusted_document_excerpt "],
        "stub": (200, "rag_retrieve_no_match.json"),
    },
    {
        "id": "cost-runbook-conflict",
        "question": "sub-a1b2c3d4 spiked on Azure. Should we buy a Log Analytics commitment "
        "tier now, according to our documents?",
        "dry_run_tool": ("find_spikes", {"threshold_pct": 200, "platform": "azure"}),
        "docs_question": "When should we buy a commitment tier after a logging cost spike?",
        "must_contain": [
            "sub-a1b2c3d4 on 2026-05-03: +331% ($2691.00)",
            "2026-05-03 to 2026-06-01 (30 days)",
            'source="runbooks/logging-cost-spike.md" section="Commitment tiers"',
            'source="programme/COST_SMOOTHING.md" section="Commitment tiers"',
            "may conflict: Commitment tiers",
            RUNBOOK_CORPUS,
        ],
        "stub": (200, "rag_retrieve_commitment_tier_conflict.json"),
    },
]


def _rag_dry_run_case(eval_case: dict) -> str:
    """Tool facts (if the case names a cost tool) followed by the stubbed docs result."""
    status, payload = eval_case["stub"]
    if isinstance(payload, str):
        payload = json.loads((FIXTURES / payload).read_text())
    with policy_docs.override(
        policy_docs.RagConfig(url="http://127.0.0.1:8765"),
        policy_docs.stub_transport(status, payload),
    ):
        docs = search_observability_docs(eval_case.get("docs_question", eval_case["question"]))
    if "dry_run_tool" not in eval_case:
        return docs
    return _dry_run_case(eval_case) + "\n" + docs


def _check_rag_case(eval_case: dict) -> tuple[bool, list[str]]:
    """must_contain as for every case, plus nothing from must_not_contain."""
    answer = _rag_dry_run_case(eval_case)
    ok, missing = _check_eval(answer, eval_case["must_contain"])
    present = [e for e in eval_case.get("must_not_contain", []) if e in answer]
    return ok and not present, missing + [f"unexpected: {e}" for e in present]


def _normalise(text: str) -> str:
    """Lowercase and treat '-', '_' and whitespace as the same separator.

    Models often rewrite identifiers ('project-alpha' -> 'Project Alpha').
    The fact is still correct, so the check should not depend on styling.
    """
    return re.sub(r"[\s_\-]+", " ", text.lower())


def _check_eval(answer: str, must_contain: list[str]) -> tuple[bool, list[str]]:
    haystack = _normalise(answer)
    missing = [e for e in must_contain if _normalise(e) not in haystack]
    return not missing, missing


def _dry_run_case(eval_case: dict) -> str:
    spec = eval_case["dry_run_tool"]
    return _dry_run_answer(spec) if isinstance(spec, list) else _dry_run_answer(*spec)


def _dry_run_answer(tool_name, tool_input: dict | None = None) -> str:
    """Run a tool directly and return its output string.

    *tool_name* may instead be a list of (name, input) pairs for questions that
    need several tools; their outputs are joined.
    """
    if isinstance(tool_name, list):
        return "\n".join(TOOL_DISPATCH[name](args) for name, args in tool_name)
    return TOOL_DISPATCH[tool_name](tool_input)


def run_evals(
    verbose: bool = False,
    *,
    dry_run: bool = False,
    provider: str | None = None,
    model: str | None = None,
) -> tuple[int, int]:
    """Run all eval cases. Returns (passed, failed) counts."""
    passed = 0
    failed = 0

    for eval_case in EVALS:
        if dry_run:
            answer = _dry_run_case(eval_case)
        else:
            kwargs = {"provider": provider} if provider else {}
            answer, _messages = run(eval_case["question"], model=model, **kwargs)

        ok, missing = _check_eval(answer, eval_case["must_contain"])
        mode = "DRY" if dry_run else "LIVE"
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] [{mode}] {eval_case['id']}")

        if not ok or verbose:
            if not dry_run:
                print(f"       Q: {eval_case['question']}")
            print(f"       A: {answer[:300]}")
            if missing:
                print(f"       Missing: {missing}")

        if ok:
            passed += 1
        else:
            failed += 1

    if dry_run:
        for eval_case in RAG_DRY_RUN_EVALS:
            ok, missing = _check_rag_case(eval_case)
            print(f"[{'PASS' if ok else 'FAIL'}] [DRY] {eval_case['id']} (stub retrieval)")
            if missing:
                print(f"       Missing: {missing}")
            passed, failed = passed + ok, failed + (not ok)

    print(f"\nResults: {passed}/{passed + failed} passed")
    return passed, failed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run obs-agent evals")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check tool outputs directly (no LLM API calls).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print details for every eval case.",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "deepseek", "openai", "ollama"],
        help="LLM provider for live evals (default: agent default).",
    )
    parser.add_argument("--model", help="Model name for live evals.")
    args = parser.parse_args()
    run_evals(
        verbose=args.verbose,
        dry_run=args.dry_run,
        provider=args.provider,
        model=args.model,
    )
