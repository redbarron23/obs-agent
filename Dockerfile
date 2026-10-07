FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
RUN uv pip install --system --no-cache -r requirements.txt

COPY agent.py api.py app.py data.py tools.py ./

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000 8501

# Default: the REST API. The compose file overrides this for the Streamlit UI.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
