"""
Eval harness for the obs-agent.

Tests a fixed set of questions against known-correct answers
(verified from the deterministic synthetic data in data.py).

Because LLM output is non-deterministic, we check for *presence of
correct facts* (must_contain) rather than exact-string matches.

Dry-run mode checks tool outputs directly — fully deterministic, no API key.
"""

from __future__ import annotations

from agent import run
from tools import TOOL_DISPATCH

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
        "question": (
            "Give me a summary of the top overage on Azure "
            "and the top cost on GCP"
        ),
        "must_contain": [
            "sub-a1b2c3d4",
            "project-alpha",
        ],
        "dry_run_tool": ("compare_cross_cloud", {}),
    },
    {
        "id": "azure-daily-trend",
        "question": (
            "What is the daily cost trend for subscription sub-a1b2c3d4?"
        ),
        "must_contain": ["sub-a1b2c3d4"],
        "dry_run_tool": (
            "get_daily_trend",
            {"platform": "azure", "identifier": "sub-a1b2c3d4"},
        ),
    },
    {
        "id": "gcp-daily-trend",
        "question": (
            "Show me the daily cost trend for project project-bravo"
        ),
        "must_contain": ["project-bravo"],
        "dry_run_tool": (
            "get_daily_trend",
            {"platform": "gcp", "identifier": "project-bravo"},
        ),
    },
    {
        "id": "find-spikes-azure",
        "question": (
            "Find any cost spikes above 200% across Azure"
        ),
        "must_contain": ["sub-a1b2c3d4"],
        "dry_run_tool": ("find_spikes", {"threshold_pct": 200}),
    },
    {
        "id": "find-spikes-gcp",
        "question": (
            "Find any cost spikes above 200% across GCP"
        ),
        "must_contain": ["project-bravo"],
        "dry_run_tool": ("find_spikes", {"threshold_pct": 200}),
    },
]


def _check_eval(answer: str, must_contain: list[str]) -> tuple[bool, list[str]]:
    missing = [e for e in must_contain if e not in answer]
    return not missing, missing


def _dry_run_answer(tool_name: str, tool_input: dict) -> str:
    """Run a tool directly and return its output string."""
    if tool_name == "compare_cross_cloud":
        return TOOL_DISPATCH[tool_name](tool_input)
    return TOOL_DISPATCH[tool_name](tool_input)


def run_evals(verbose: bool = False, *, dry_run: bool = False) -> tuple[int, int]:
    """Run all eval cases. Returns (passed, failed) counts."""
    passed = 0
    failed = 0

    for eval_case in EVALS:
        if dry_run:
            tool_name, tool_input = eval_case["dry_run_tool"]
            answer = _dry_run_answer(tool_name, tool_input)
        else:
            answer, _messages = run(eval_case["question"])

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
        "-v", "--verbose",
        action="store_true",
        help="Print details for every eval case.",
    )
    args = parser.parse_args()
    run_evals(verbose=args.verbose, dry_run=args.dry_run)
