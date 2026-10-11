"""
Tests for provider/model configuration (issue #5): OBS_AGENT_PROVIDER /
OBS_AGENT_MODEL, the Claude adaptive-thinking request profile, verbatim
replay of Anthropic content blocks, refusal handling, and "answered by"
reporting across the CLI, API and UI. No live API calls.
"""

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from anthropic.types.beta import BetaTextBlock, BetaThinkingBlock, BetaToolUseBlock

import agent
from agent import ConfigError

ROOT = Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("OBS_AGENT_PROVIDER", raising=False)
    monkeypatch.delenv("OBS_AGENT_MODEL", raising=False)
    monkeypatch.delenv("OBS_AGENT_MAX_TOKENS", raising=False)


def _response(blocks, stop_reason="end_turn", model="claude-opus-5-5", stop_details=None):
    return SimpleNamespace(
        content=blocks, stop_reason=stop_reason, model=model, stop_details=stop_details
    )


class TestDefaults:
    def test_unset_is_ollama(self):
        assert agent.check_config() == ("ollama", "qwen2.5:7b")

    def test_provider_from_env(self, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_PROVIDER", "Anthropic ")
        assert agent.check_config() == ("anthropic", "claude-opus-5-5")

    def test_model_override_applies_to_default_provider_only(self, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_PROVIDER", "openai")
        monkeypatch.setenv("OBS_AGENT_MODEL", "gpt-custom")
        assert agent.default_model() == "gpt-custom"
        assert agent.default_model("openai") == "gpt-custom"
        assert agent.default_model("anthropic") == "claude-opus-5-5"

    def test_invalid_provider_fails_with_clear_message(self, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_PROVIDER", "bard")
        with pytest.raises(ConfigError, match="not a supported provider.*anthropic"):
            agent.check_config()

    def test_run_uses_env_default(self, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_PROVIDER", "openai")
        seen = {}

        def fake_provider(name, model):
            seen.update(name=name, model=model)
            raise RuntimeError("stop")

        with patch("agent.Provider", side_effect=fake_provider), pytest.raises(RuntimeError):
            agent.run("q")
        assert seen == {"name": "openai", "model": "gpt-5.4-mini"}

    def test_warning_only_for_small_local_model(self):
        assert "small local model" in agent.model_warning("ollama", "qwen2.5:7b")
        assert agent.model_warning("anthropic", "claude-opus-5-5") is None
        assert agent.model_warning("openai", "gpt-5.4-mini") is None


class TestCLI:
    def test_invalid_env_exits(self, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_PROVIDER", "bard")
        monkeypatch.setattr("sys.argv", ["agent.py", "-q", "hi"])
        with pytest.raises(SystemExit, match="OBS_AGENT_PROVIDER"):
            agent.main()

    def test_flag_beats_env(self, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_PROVIDER", "anthropic")
        monkeypatch.setattr("sys.argv", ["agent.py", "-q", "hi", "--provider", "openai"])
        with patch("agent.run", return_value=("ok", [])) as run:
            agent.main()
        assert run.call_args.kwargs["provider"] == "openai"
        assert run.call_args.kwargs["model"] == "gpt-5.4-mini"


class TestAnthropicRequest:
    def _provider(self, model):
        p = agent.Provider.__new__(agent.Provider)
        p.name, p.model, p._client = "anthropic", model, MagicMock()
        return p

    def test_adaptive_model_profile(self):
        p = self._provider("claude-opus-5-5")
        p.create([{"role": "user", "content": "hi"}])
        kwargs = p._client.beta.messages.create.call_args.kwargs
        assert kwargs["max_tokens"] == 16000
        assert kwargs["output_config"] == {"effort": "medium"}
        assert kwargs["thinking"]["type"] == "adaptive"
        assert kwargs["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
        assert kwargs["fallbacks"] == "default"
        assert set(kwargs["betas"]) == {
            "thinking-binding-controls-2026-08-01",
            "server-side-fallback-2026-07-01",
        }
        assert "temperature" not in kwargs and "tool_choice" not in kwargs
        p._client.messages.create.assert_not_called()

    def test_older_model_keeps_plain_request(self):
        p = self._provider("claude-sonnet-4-6")
        p.create([{"role": "user", "content": "hi"}])
        kwargs = p._client.messages.create.call_args.kwargs
        # No thinking on this path, so the whole budget goes to the answer.
        assert kwargs["max_tokens"] == agent.DEFAULT_MAX_TOKENS == 4096
        assert "thinking" not in kwargs and "betas" not in kwargs

    def test_max_tokens_env_override(self, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_MAX_TOKENS", "8192")
        p = self._provider("claude-sonnet-4-6")
        p.create([{"role": "user", "content": "hi"}])
        assert p._client.messages.create.call_args.kwargs["max_tokens"] == 8192

    def test_invalid_max_tokens_override_is_a_config_error(self, monkeypatch):
        for bad in ("0", "-1", "lots"):
            monkeypatch.setenv("OBS_AGENT_MAX_TOKENS", bad)
            with pytest.raises(agent.ConfigError):
                agent.max_tokens_setting()

    def test_adaptive_profile_ignores_override(self):
        # The adaptive path reserves its own budget for thinking; the shared
        # answer budget must not shrink it.
        p = self._provider("claude-opus-5-5")
        p.create([{"role": "user", "content": "hi"}])
        assert p._client.beta.messages.create.call_args.kwargs["max_tokens"] == 16000

    def test_streaming_uses_same_profile(self):
        p = self._provider("claude-opus-5-5")
        p.create([{"role": "user", "content": "hi"}], stream=True)
        assert p._client.beta.messages.stream.call_args.kwargs["fallbacks"] == "default"


class TestTruncation:
    """A response cut off by the token cap must say so, not look complete."""

    def _patch(self, monkeypatch, stop_reason, text="partial answ"):
        """Make Provider.create() return a capped response, keeping the class intact."""
        block = SimpleNamespace(type="text", text=text)
        response = SimpleNamespace(content=[block], stop_reason=stop_reason, model="m", usage=None)
        monkeypatch.setattr(agent.Provider, "__init__", lambda self, *a, **k: None)
        monkeypatch.setattr(agent.Provider, "create", lambda self, *a, **k: response)

    def _run(self, on_token=None):
        return agent._run_loop(
            "q",
            provider="anthropic",
            model="claude-sonnet-4-6",
            verbose=False,
            stream=False,
            messages=None,
            on_token=on_token,
        )

    def test_max_tokens_stop_reason_is_surfaced(self, monkeypatch):
        self._patch(monkeypatch, "max_tokens")
        answer, messages = self._run()
        assert answer.startswith("partial answ")
        assert "truncated" in answer.lower()
        assert "OBS_AGENT_MAX_TOKENS" in answer
        # The note is part of the recorded turn, so follow-ups see it too.
        assert "truncated" in messages[-1]["content"].lower()

    def test_end_turn_is_not_flagged_as_truncated(self, monkeypatch):
        self._patch(monkeypatch, "end_turn")
        answer, _ = self._run()
        assert answer == "partial answ"
        assert "truncated" not in answer.lower()

    def test_truncation_with_no_text_still_reports(self, monkeypatch):
        self._patch(monkeypatch, "max_tokens", text="")
        answer, _ = self._run()
        assert answer.lower().startswith("[response truncated")

    def test_note_is_streamed_to_the_callback(self, monkeypatch):
        # On the streaming path a callback is attached, so the note reaches it
        # too rather than appearing only at the end of the response.
        self._patch(monkeypatch, "max_tokens")
        seen: list[str] = []
        monkeypatch.setattr(
            agent.Provider,
            "stream_response",
            staticmethod(
                lambda *a, **k: (
                    SimpleNamespace(
                        content=[SimpleNamespace(type="text", text="partial answ")],
                        stop_reason="max_tokens",
                        model="m",
                        usage=None,
                    ),
                    ["partial answ"],
                    [],
                )
            ),
        )
        agent._run_loop(
            "q",
            provider="anthropic",
            model="claude-sonnet-4-6",
            verbose=False,
            stream=True,
            messages=None,
            on_token=seen.append,
        )
        assert any("truncated" in t.lower() for t in seen)

    def test_openai_length_finish_reason_normalises_to_max_tokens(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="length")], stop_reason=None
        )
        del response.stop_reason  # force the OpenAI branch
        assert agent.Provider.stop_reason(response) == "max_tokens"


class TestReplay:
    def _tool_response(self):
        return _response(
            [
                BetaThinkingBlock(type="thinking", thinking="", signature="sig-1"),
                BetaToolUseBlock(
                    type="tool_use",
                    id="toolu_1",
                    name="suggest_labels",
                    input={"resource_id": "svc-ingest"},
                ),
            ],
            stop_reason="tool_use",
        )

    def test_thinking_blocks_are_kept_and_replayed_verbatim(self):
        msg = agent._parse_assistant_response("anthropic", self._tool_response())
        assert msg["anthropic_content"][0] == {
            "type": "thinking",
            "thinking": "",
            "signature": "sig-1",
        }
        assert msg["tool_calls"][0]["name"] == "suggest_labels"
        assert msg["model"] == "claude-opus-5-5"
        history = [{"role": "user", "content": "q"}, msg]
        formatted = agent._format_messages_for_anthropic(history)
        assert formatted[1] == {"role": "assistant", "content": msg["anthropic_content"]}

    def test_openai_formatter_strips_internal_keys(self):
        final = agent._final_message(
            "anthropic", _response([BetaTextBlock(type="text", text="done")]), "done"
        )
        assert "anthropic_content" in final
        out = agent._format_messages_for_openai([{"role": "user", "content": "q"}, final])
        assert out[1] == {"role": "assistant", "content": "done"}

    def test_full_loop_replays_thinking_on_second_call(self):
        final = _response([BetaTextBlock(type="text", text="data-platform (60%)")])
        sent = []

        def fake_create(messages, stream=False):
            sent.append(agent._format_messages_for_anthropic(messages))
            return self._tool_response() if len(sent) == 1 else final

        with (
            patch("agent.Provider.create", side_effect=fake_create),
            patch("agent.Provider.__init__", return_value=None),
        ):
            answer, history = agent.run("team of svc-ingest?", provider="anthropic")
        assert answer == "data-platform (60%)"
        assert sent[1][1]["content"][0]["type"] == "thinking"
        assert agent.served_by(history) == "claude-opus-5-5"


class TestRefusal:
    def test_refusal_returns_explanation(self):
        resp = _response([], stop_reason="refusal", stop_details=SimpleNamespace(category="cyber"))
        with (
            patch("agent.Provider.create", return_value=resp),
            patch("agent.Provider.__init__", return_value=None),
        ):
            answer, history = agent.run("q", provider="anthropic")
        assert "declined" in answer and "cyber" in answer
        assert history[-1]["role"] == "assistant"

    def test_fallback_model_reported(self):
        resp = _response([BetaTextBlock(type="text", text="answer")], model="claude-opus-5")
        with (
            patch("agent.Provider.create", return_value=resp),
            patch("agent.Provider.__init__", return_value=None),
        ):
            _, history = agent.run("q", provider="anthropic", model="claude-opus-5-5")
        assert agent.served_by(history) == "claude-opus-5"


class TestAPI:
    @pytest.fixture
    def client(self, monkeypatch):
        from fastapi.testclient import TestClient

        import api

        monkeypatch.setenv("OBS_AGENT_API_KEY", "test-api-key")
        api.sessions = api.SessionStore()
        api.agent_limiter = api.AgentRequestLimiter(
            api.MAX_AGENT_REQUESTS_PER_MINUTE, api.MAX_CONCURRENT_AGENT_REQUESTS
        )
        return TestClient(api.app, headers={"X-API-Key": "test-api-key"})

    def test_providers_reflect_env(self, client, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_PROVIDER", "anthropic")
        body = client.get("/providers").json()
        assert body["default"] == "anthropic"
        assert body["default_model"] == "claude-opus-5-5"
        assert body["warning"] is None

    def test_ask_uses_env_default_and_reports(self, client, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_PROVIDER", "anthropic")
        history = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "model": "claude-opus-5"},
        ]
        with patch("api.run", return_value=("a", history)) as run:
            body = client.post("/ask", json={"question": "q"}).json()
        assert run.call_args.kwargs["provider"] == "anthropic"
        assert body["provider"] == "anthropic"
        assert body["model"] == "claude-opus-5-5"
        assert body["served_by"] == "claude-opus-5"
        assert body["warning"] is None

    def test_ask_warns_for_local_model(self, client):
        with patch("api.run", return_value=("a", [{"role": "user", "content": "q"}])):
            body = client.post("/ask", json={"question": "q"}).json()
        assert body["provider"] == "ollama"
        assert "small local model" in body["warning"]

    def test_unknown_provider_in_request(self, client):
        assert client.post("/ask", json={"question": "q", "provider": "bard"}).status_code == 422

    def test_invalid_env_fails_at_startup(self, monkeypatch):
        import api

        monkeypatch.setenv("OBS_AGENT_PROVIDER", "bard")
        with pytest.raises(RuntimeError, match="OBS_AGENT_PROVIDER"):
            importlib.reload(api)
        monkeypatch.delenv("OBS_AGENT_PROVIDER")
        importlib.reload(api)


class TestUI:
    def _app(self):
        from streamlit.testing.v1 import AppTest

        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)

    def test_dropdown_follows_env(self, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_PROVIDER", "openai")
        at = self._app().run()
        assert not at.exception
        assert at.sidebar.selectbox[0].value == "openai"
        assert not at.sidebar.warning

    def test_local_model_warning(self):
        at = self._app().run()
        assert at.sidebar.selectbox[0].value == "ollama"
        assert "small local model" in at.sidebar.warning[0].value

    def test_invalid_env_shows_error(self, monkeypatch):
        monkeypatch.setenv("OBS_AGENT_PROVIDER", "bard")
        at = self._app().run()
        assert at.error and "OBS_AGENT_PROVIDER" in at.error[0].value
