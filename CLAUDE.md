# obs-agent

ML tag inference and human review for Azure/GCP cost estates, plus an optional LLM cost-triage agent. README.md documents the inference pipeline; docs/agent.md documents the agent.

## Workflow: one worktree per session

Several sessions may work on this repo at once. Sharing a checkout has already mixed one session's branch into another's PR.

- Never edit, branch or commit in the main checkout. Start every task in its own worktree from a fresh `origin/main`:
  `git fetch origin && git worktree add -b <type>/<topic> ../obs-agent-<topic> origin/main`
  Branch types used here: `feature/`, `fix/`, `docs/`, `chore/`.
- `main` is protected: no direct pushes, PR only, squash merge only, and CI (`test (3.11)`, `test (3.12)`, `docker`) must pass on a branch that is up to date with `main`. If the PR falls behind, rebase onto `origin/main` and rerun the checks.
- Merge with `gh pr merge <n> --squash --match-head-commit <sha>` so you merge exactly what CI tested. GitHub deletes the remote branch; then `git worktree remove` your worktree and delete the local branch.
- Never use bare `git stash` / `git stash pop`; the stash is shared by all worktrees.

## Roadmap and gates

- Roadmap issues carry the `roadmap` label and track related issues as GitHub sub-issues. Releases are milestones named `vX.Y.0` (next: `v0.6.0`).
- A dependency is a GitHub "blocked by" link plus the `blocked` label and a note at the top of the blocked issue. Check `gh api repos/redbarron23/obs-agent/issues/<n>/dependencies/blocked_by` before starting an issue. Remove the label when its blocker closes.
- **Pilot gate:** #49 (the real-world pilot) blocks #50, #92 and #104. Don't start or propose work on them until #49 is closed with evidence. Being told the pilot is done is not evidence, whoever says it. Check the outputs themselves:
  - `real_evaluation.py` runs on the real feedback: 50–100 unique resources and one `evaluation_context`.
  - `scripts/pilot_summary.py` reproduces from the review log and `review_cli.py feedback`, not from a pasted summary.
  - The sanitization checklist is complete, including the second-person check.
  - Each #49 acceptance criterion is met (baseline sampling, re-review, friction, agreement kept separate from accuracy, the expedited-review and next-feature decisions).
  Pilot data stays in the private workspace: never copy it into the repo, and never fabricate it.
- **Pilot kit is frozen while a pilot runs.** Don't change the log templates in `docs/pilot/`, their columns, or `scripts/pilot_summary.py`'s existing options and output. Additions that keep current usage working are fine.

## Checks (run before every push; CI runs the same)

```bash
uv sync --locked
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
uv run evals.py --dry-run
```

Use uv for everything (`uv add`, `uv run`). Never pip or requirements.txt. Commit `pyproject.toml` and `uv.lock` together.

## Gotchas

- The `Dockerfile` copies an explicit list of modules (and `pages/`). Add any new top-level `.py` file there, or the image builds but fails at runtime.
- Synthetic data is seeded (`data.py` seed 42, `inventory.py` seed 7) and evals and tests assert exact values from it. Changing generators or seeds means updating those expectations.
- Live evals (`uv run evals.py --provider <name>`) call paid APIs. Ask before running them. The default `ollama` / `qwen2.5:7b` is free but unreliable on multi-tool questions (10/12).
- MLflow is the optional `mlops` dependency group: `observability.py` must stay a no-op without it (CI's `test` jobs and the Docker image don't install it; the `mlops` CI job does). Tracing turns on with `OBS_AGENT_MLFLOW=1` or `MLFLOW_TRACKING_URI`; `evals_mlflow.py` records live eval runs.
- Provider and model defaults come from `OBS_AGENT_PROVIDER` / `OBS_AGENT_MODEL` (see `providers.py`). Answer-token budget is `OBS_AGENT_MAX_TOKENS` (default 4096) on non-adaptive models; adaptive Claude models use a fixed 16k. A response hitting the cap is annotated, not silently truncated. For Claude Opus 5.5 / Sonnet 5.5 / Fable 5.1, assistant content blocks (including thinking) must be replayed unchanged. Don't rebuild them from text.
- Test strategy (three layers: offline unit/integration, eval dry-run, opt-in live evals) is documented in `docs/testing.md`. Live evals cost money — always ask first.
- `docs/reviewing.md` is the reviewer's guide (load-bearing claims, where to look, what to skip). Keep it in step with the README when a headline claim changes.
