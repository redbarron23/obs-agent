# obs-agent

ML tag inference and human review for Azure/GCP cost estates, plus an optional LLM cost-triage agent. README.md documents the inference pipeline; docs/agent.md documents the agent.

## Workflow: one worktree per session

Several sessions may work on this repo at once. Sharing a checkout has already mixed one session's branch into another's PR.

- Never edit, branch or commit in the main checkout. Start every task in its own worktree from a fresh `origin/main`:
  `git fetch origin && git worktree add -b <type>/<topic> ../obs-agent-<topic> origin/main`
  Branch types used here: `feature/`, `fix/`, `docs/`, `chore/`.
- `main` is protected: no direct pushes, PR only, squash merge only, and CI (`test (3.10)`, `test (3.11)`, `test (3.12)`, `docker`) must pass on a branch that is up to date with `main`. If the PR falls behind, rebase onto `origin/main` and rerun the checks.
- Merge with `gh pr merge <n> --squash --match-head-commit <sha>` so you merge exactly what CI tested. GitHub deletes the remote branch; then `git worktree remove` your worktree and delete the local branch.
- Never use bare `git stash` / `git stash pop`; the stash is shared by all worktrees.

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
- Provider and model defaults come from `OBS_AGENT_PROVIDER` / `OBS_AGENT_MODEL` (see `agent.py`). Answer-token budget is `OBS_AGENT_MAX_TOKENS` (default 4096) on non-adaptive models; adaptive Claude models use a fixed 16k. A response hitting the cap is annotated, not silently truncated. For Claude Opus 5.5 / Sonnet 5.5 / Fable 5.1, assistant content blocks (including thinking) must be replayed unchanged. Don't rebuild them from text.
- Test strategy (three layers: offline unit/integration, eval dry-run, opt-in live evals) is documented in `docs/testing.md`. Live evals cost money — always ask first.
