"""Tests for the FastAPI service (LLM mocked)."""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import api
from api import app


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("OBS_AGENT_API_KEY", raising=False)
    api.sessions = api.SessionStore()


@pytest.fixture
def client():
    return TestClient(app)


def _history(answer="ok", tool="get_azure_top_overages"):
    return [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "1", "name": tool, "input": {}}]},
        {"role": "tool", "tool_call_id": "1", "content": "data"},
        {"role": "assistant", "content": answer},
    ]


class TestBasics:
    def test_health(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_providers(self, client):
        body = client.get("/providers").json()
        assert body["default"] == "ollama"
        assert "openai" in body["models"]


class TestAsk:
    def test_ask_returns_answer_and_tools(self, client):
        with patch("api.run", return_value=("$18,411", _history())) as m:
            r = client.post("/ask", json={"question": "top overage?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "$18,411"
        assert body["tools_called"] == ["get_azure_top_overages"]
        assert body["provider"] == "ollama"
        assert body["session_id"]
        assert m.call_args.kwargs["messages"] is None

    def test_session_memory_reused(self, client):
        with patch("api.run", return_value=("a", _history())) as m:
            sid = client.post("/ask", json={"question": "one"}).json()["session_id"]
            client.post("/ask", json={"question": "two", "session_id": sid})
        assert m.call_args.kwargs["messages"] == _history()

    def test_unknown_provider_422(self, client):
        r = client.post("/ask", json={"question": "x", "provider": "nope"})
        assert r.status_code == 422

    def test_empty_question_422(self, client):
        assert client.post("/ask", json={"question": ""}).status_code == 422

    def test_backend_error_502(self, client):
        with patch("api.run", side_effect=RuntimeError("boom")):
            r = client.post("/ask", json={"question": "x"})
        assert r.status_code == 502
        assert "boom" in r.json()["detail"]


class TestStream:
    def test_stream_events(self, client):
        def fake_stream(question, **kw):
            kw["result_holder"].update(answer="ab", messages=_history("ab"))
            yield "a"
            yield "b"

        with patch("api.stream_run", fake_stream):
            r = client.post("/ask/stream", json={"question": "x"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert 'data: {"token": "a"}' in r.text
        assert "event: done" in r.text
        done = json.loads(r.text.split("event: done\ndata: ")[1].split("\n")[0])
        assert done["tools_called"] == ["get_azure_top_overages"]

    def test_stream_error_event(self, client):
        def failing(question, **kw):
            raise RuntimeError("bad key")
            yield  # pragma: no cover

        with patch("api.stream_run", failing):
            r = client.post("/ask/stream", json={"question": "x"})
        assert "event: error" in r.text
        assert "bad key" in r.text


class TestAuth:
    def test_key_required_when_configured(self, client, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_API_KEY", "secret")
        assert client.get("/providers").status_code == 401
        assert client.get("/providers", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/providers", headers={"X-API-Key": "secret"}).status_code == 200

    def test_health_stays_open(self, client, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_API_KEY", "secret")
        assert client.get("/health").status_code == 200


class TestSessions:
    def test_delete(self, client):
        with patch("api.run", return_value=("a", _history())):
            sid = client.post("/ask", json={"question": "x"}).json()["session_id"]
        assert client.delete(f"/sessions/{sid}").status_code == 200
        assert client.delete(f"/sessions/{sid}").status_code == 404

    def test_lru_eviction(self):
        store = api.SessionStore(max_sessions=2)
        store.put("a", [1]); store.put("b", [2]); store.put("c", [3])
        assert store.get("a") is None and store.get("c") == [3]


def test_stream_run_propagates_errors():
    from agent import stream_run

    with patch("agent.run", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            list(stream_run("q"))


class TestOpenAPI:
    def test_errors_and_sse_documented(self, client):
        spec = client.get("/openapi.json").json()
        ask = spec["paths"]["/ask"]["post"]["responses"]
        assert {"200", "401", "422", "502"} <= set(ask)
        stream = spec["paths"]["/ask/stream"]["post"]
        assert "text/event-stream" in stream["responses"]["200"]["content"]
        assert "event: done" in stream["description"]

    def test_request_example_present(self, client):
        schema = client.get("/openapi.json").json()["components"]["schemas"]["AskRequest"]
        assert schema["examples"][0]["question"]
