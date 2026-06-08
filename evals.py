"""
Eval harness for the obs-agent.

Tests a fixed set of questions against known-correct answers
(verified from the deterministic synthetic data in data.py).

Because LLM output is non-deterministic, we check for *presence of
correct facts* (must_contain) rather than exact-string matches.
"""

from agent import run

# Ground-truth expected identifiers.  These are stable because data.py
# uses a fixed random seed.
EVALS = [
    {
        "id": "azure-top-overage",
        "question": "Which Azure subscription has the highest overage cost?",
        "must_contain": ["sub-a1b2c3d4"],
    },
    {
        "id": "gcp-top-project",
        "question": "Which GCP project has the highest logging cost?",
        "must_contain": ["project-alpha"],
    },
    {
        "id": "gcp-top-3",
        "question": "Show me the top 3 most expensive GCP projects",
        "must_contain": [
            "project-alpha",
            "project-bravo",
        ],
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
    },
    {
        "id": "azure-daily-trend",
        "question": (
            "What is the daily cost trend for subscription sub-a1b2c3d4?"
        ),
        "must_contain": ["sub-a1b2c3d4"],
    },
    {
        "id": "gcp-daily-trend",
        "question": (
            "Show me the daily cost trend for project project-bravo"
        ),
        "must_contain": ["project-bravo"],
    },
    {
        "id": "find-spikes-azure",
        "question": (
            "Find any cost spikes above 200% across Azure"
        ),
        "must_contain": ["sub-a1b2c3d4"],
    },
    {
        "id": "find-spikes-gcp",
        "question": (
            "Find any cost spikes above 200% across GCP"
        ),
        "must_contain": ["project-bravo"],
    },
]


def run_evals(verbose: bool = False) -> None:
    passed = 0
    failed = 0

    for eval_case in EVALS:
        answer = run(eval_case["question"])
        ok = all(
            expected in answer
            for expected in eval_case["must_contain"]
        )

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {eval_case['id']}")

        if not ok or verbose:
            print(f"       Q: {eval_case['question']}")
            print(f"       A: {answer[:300]}")
            missing = [
                e for e in eval_case["must_contain"]
                if e not in answer
            ]
            if missing:
                print(f"       Missing: {missing}")

        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\nResults: {passed}/{passed + failed} passed")


if __name__ == "__main__":
    run_evals(verbose=False)
