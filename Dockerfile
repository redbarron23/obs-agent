FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install the exact locked runtime deps (no dev group) into /app/.venv.
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev
ENV PATH="/app/.venv/bin:$PATH"

COPY agent.py api.py app.py data.py drift.py inventory.py inventory_import.py labeling.py label_review.py label_config.py review_context.py review_workspaces.py observability.py plan_validation.py providers.py real_evaluation.py tools.py ./
COPY pages/ pages/
COPY fixtures/ fixtures/

# Review decisions live in /app/state; compose mounts a shared volume here.
RUN useradd --create-home appuser && mkdir /app/state && chown appuser /app/state
ENV OBS_AGENT_DECISIONS=/app/state/label_decisions.db \
    OBS_AGENT_REVIEW_DIR=/app/state/reviews
USER appuser

EXPOSE 8000 8501

# Default: the REST API. The compose file overrides this for the Streamlit UI.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
