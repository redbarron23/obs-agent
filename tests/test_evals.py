"""Tests for the eval harness."""

import pytest

from evals import EVALS, _check_eval, run_evals


class TestEvalDryRun:
    """Deterministic eval checks against tool output (no LLM)."""

    def test_all_dry_run_pass(self):
        passed, failed = run_evals(dry_run=True)
        assert failed == 0
        assert passed == len(EVALS)

    @pytest.mark.parametrize("eval_case", EVALS, ids=[e["id"] for e in EVALS])
    def test_each_dry_run_case(self, eval_case):
        from evals import _dry_run_answer

        tool_name, tool_input = eval_case["dry_run_tool"]
        answer = _dry_run_answer(tool_name, tool_input)
        ok, missing = _check_eval(answer, eval_case["must_contain"])
        assert ok, f"Missing {missing} in: {answer[:200]}"


@pytest.mark.integration
class TestEvalLive:
    """Live LLM evals — skipped unless ANTHROPIC_API_KEY is set."""

    @pytest.fixture(autouse=True)
    def require_api_key(self):
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

    def test_live_evals(self):
        passed, failed = run_evals(dry_run=False)
        assert failed == 0, f"{failed} live eval(s) failed"
