# Interfaces

obs-agent has three ways in: a Streamlit web UI, a REST API, and a Docker Compose stack. The label
review flow needs no LLM in any of them — only the optional cost Q&A agent does.

- [Web UI (Streamlit)](#web-ui-streamlit)
- [REST API (FastAPI)](#rest-api-fastapi)
- [Docker](#docker)
- [Hosted demo (Streamlit Community Cloud)](#hosted-demo-streamlit-community-cloud)

## Web UI (Streamlit)

```bash
uv run streamlit run app.py
```

The sidebar has two pages. **Label review** is the main one: the review queue, per-row decisions
and the export buttons described in the README. The **chat** page drives the cost Q&A agent with
provider/model selection, conversation history and example prompts ([docs/agent.md](agent.md)).

## REST API (FastAPI)

Both the label review and the agent are available as a standalone HTTP service, so other apps
(dashboards, chat bots, pipelines) can call them without importing any Python:

```bash
export OBS_AGENT_API_KEY="$(openssl rand -hex 32)"
uv run uvicorn api:app --reload          # http://localhost:8000/docs for interactive OpenAPI docs
```

### Label endpoints

The ones the review UI is built on, and the ones that need no LLM:

| Endpoint | Purpose |
|---|---|
| `GET /labels/coverage` | Untagged / suspect / auto-apply-ready counts per label |
| `GET /resources` | Review queue; filters `label`, `status`, `cloud`, `min_confidence`, `max_confidence`, `search`, plus `limit` / `offset` |
| `GET /resources/{id}/labels` | Current tags, suggestions, confidence, reasons and alternatives for one resource |
| `PUT` / `DELETE /resources/{id}/labels/{label}/decision` | Record (`accept` / `reject` / `override` + `value`) or undo a review decision |
| `GET /labels/decisions` | All recorded decisions |
| `GET /labels/export?format=csv\|sh\|tfvars` | Resolved changes as a download; `include_auto=false` exports reviewed decisions only |

### Agent endpoints

See [docs/agent.md](agent.md) for request/response detail and the SSE streaming format:

| Endpoint | Purpose |
|---|---|
| `POST /ask` | `{"question", "provider"?, "model"?, "session_id"?}` → `{"answer", "session_id", "tools_called", ...}` |
| `POST /ask/stream` | Same request, answer streamed as server-sent events (`token` events, then `done` with `tools_called`, or `error`) |
| `GET /providers` | Supported providers and default models |
| `GET /health` | Liveness probe (always open) |
| `DELETE /sessions/{id}` | Forget a conversation |

```bash
curl -X PUT localhost:8000/resources/svc-ingest/labels/team/decision \
  -H "X-API-Key: $OBS_AGENT_API_KEY" \
  -H 'content-type: application/json' -d '{"decision": "accept"}'

curl -s localhost:8000/ask -H 'content-type: application/json' \
  -H "X-API-Key: $OBS_AGENT_API_KEY" \
  -d '{"question": "Which Azure subscription has the highest overage?", "session_id": "demo"}'
```

### Errors

| Status | Meaning |
|---|---|
| `401` | Missing or wrong `X-API-Key` |
| `429` | More than 20 agent requests/minute from one directly connected client, or more than 4 agent requests are running |
| `503` | `OBS_AGENT_API_KEY` is not configured (except `/health`) |
| `404` | `DELETE /sessions/{id}` for an unknown session |
| `422` | Invalid body (empty question, over 4000 chars) or unknown provider |
| `502` | The LLM backend failed: bad key, no credits, Ollama not running |

All error bodies are `{"detail": "..."}`. Interactive docs with schemas and examples are at `/docs`
(Swagger UI) and `/redoc`; the raw spec is at `/openapi.json`.

### Running it in production

- **Sessions are in memory** (LRU, 200 max) and lost on restart. Run a single replica, or put
  sticky sessions (by `session_id`) in front. For multiple replicas, move `SessionStore` to Redis
  ([issue #16](https://github.com/redbarron23/obs-agent/issues/16)).
- **Set `OBS_AGENT_API_KEY`** before starting the API. Every route except `/health` rejects
  requests until it is configured. Docker Compose also binds both ports to loopback only; put an
  authenticated reverse proxy in front to expose either service.
- **Behind a gateway** (AWS API Gateway, Azure APIM, nginx): the service is plain HTTP/JSON plus
  SSE. Disable response buffering and use a long idle timeout on `/ask/stream`, or tokens arrive
  in one burst. Local models can take a minute for a multi-tool question.
- **Provider keys** go in environment variables (`.env` with Compose, a secret store elsewhere),
  never in requests.
- **Request size and cost:** questions are capped at 4000 characters. The API additionally allows
  20 `/ask` requests per minute per directly connected client and four concurrent agent requests;
  keep a stricter distributed limit at the gateway when running multiple replicas.

## Docker

```bash
cp .env.example .env              # add keys for hosted providers (optional)
ollama serve                      # the cost agent's default provider runs on the host
docker compose up --build         # API on :8000, Streamlit UI on :8501
```

Containers reach the host's Ollama through `host.docker.internal` (override with `OLLAMA_HOST`).
The image installs dependencies with uv, runs as a non-root user, and the API has a health check.

## Hosted demo (Streamlit Community Cloud)

Live at **https://obs-agent.streamlit.app** (Label review: [/Label_review](https://obs-agent.streamlit.app/Label_review)).
The Label review page needs no LLM, so it can be hosted for free:

1. Open [share.streamlit.io](https://share.streamlit.io) → **Create app** → repository
   `redbarron23/obs-agent`, branch `main`, main file `app.py`.
2. **Advanced settings**: Python **3.12**, and under Secrets add `OBS_AGENT_DEMO = "1"`.
3. Deploy. Dependencies install from `uv.lock`.

With `OBS_AGENT_DEMO=1` each visitor's decisions are private to their browser session and
discarded when it ends, and the chat page is switched off with a pointer to Label review (there
is no LLM in the demo).
