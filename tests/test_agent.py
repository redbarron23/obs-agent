"""
Unit tests for the agent loop in agent.py.

These tests mock the Provider class so no real LLM API calls are made.
We control exactly what the "model" returns — tool calls or final text —
and verify that the loop dispatches correctly, handles streaming,
and maintains conversation memory.
"""

from unittest.mock import patch, MagicMock, PropertyMock
from types import SimpleNamespace
import pytest


# ── Helpers ────────────────────────────────────────────────────────────


def make_anthropic_response(
    *,
    text: str | None = None,
    tool_name: str | None = None,
    tool_input: dict | None = None,
    tool_id: str = "toolu_abc123",
) -> MagicMock:
    """Build a fake Anthropic messages API response."""
    content = []
    if text:
        tblock = MagicMock()
        tblock.type = "text"
        tblock.text = text
        content.append(tblock)
    if tool_name:
        tblock = MagicMock()
        tblock.type = "tool_use"
        tblock.name = tool_name
        tblock.input = tool_input or {}
        tblock.id = tool_id
        content.append(tblock)

    resp = MagicMock()
    resp.stop_reason = "end_turn" if not tool_name else "tool_use"
    resp.content = content
    # Ensure the mocked name attribute returns the real string, not a MagicMock
    for block in content:
        if hasattr(block, "name"):
            type(block).name = PropertyMock(return_value=block.name)
    return resp


def make_openai_response(
    *,
    text: str | None = None,
    tool_name: str | None = None,
    tool_input: dict | None = None,
    tool_id: str = "call_abc123",
):
    """Build a fake OpenAI-compatible chat completion response.

    Returns a response that looks like an OpenAI ChatCompletion
    (no MagicMock, so hasattr checks work correctly).
    """
    import json

    # Build message content
    if tool_name:
        tc = SimpleNamespace()
        tc.id = tool_id
        tc.function = SimpleNamespace()
        tc.function.name = tool_name
        tc.function.arguments = json.dumps(tool_input or {})
        tc.type = "function"

        # Build the message with tool_calls
        msg = SimpleNamespace()
        msg.content = text
        msg.tool_calls = [tc]
    else:
        msg = SimpleNamespace()
        msg.content = text
        msg.tool_calls = None

    choice = SimpleNamespace()
    choice.message = msg
    choice.finish_reason = "stop" if not tool_name else "tool_calls"

    resp = SimpleNamespace()
    resp.choices = [choice]
    return resp


# ── Tool dispatch mock ─────────────────────────────────────────────────

_FAKE_TOOL_RESULTS = {
    "get_azure_top_overages": "FAKE: sub-a is top at $5000",
    "get_gcp_top_projects": "FAKE: proj-alpha is top at $8000",
}


@pytest.fixture(autouse=True)
def patch_tool_dispatch():
    """Replace real tool dispatch with fake results for testing."""
    import agent as agent_mod
    fake_dispatch = {
        name: (lambda args, r=v: r)
        for name, v in _FAKE_TOOL_RESULTS.items()
    }
    with patch.dict(agent_mod.TOOL_DISPATCH, fake_dispatch, clear=True):
        yield


# ── Tests ──────────────────────────────────────────────────────────────


class TestAgentLoop:
    """Tests for the core agent loop logic."""

    def test_simple_answer_no_tools(self):
        """Model answers directly without calling any tools."""
        fake_resp = make_anthropic_response(text="The top subscription is sub-a.")

        with patch("agent.Provider.create", return_value=fake_resp):
            from agent import run
            answer, messages = run("Which is top?")

        assert answer == "The top subscription is sub-a."
        assert len(messages) == 2  # user question + assistant answer

    def test_single_tool_call_then_answer(self):
        """Model calls one tool, gets result, then answers."""
        tool_resp = make_anthropic_response(
            tool_name="get_azure_top_overages",
            tool_input={"n": 1},
        )
        final_resp = make_anthropic_response(text="sub-a is top at $5000")

        mock_create = MagicMock(side_effect=[tool_resp, final_resp])
        with patch("agent.Provider.create", mock_create):
            from agent import run
            answer, messages = run("Which Azure sub is top?")

        assert "sub-a" in answer
        assert "5000" in answer
        assert mock_create.call_count == 2

    def test_multiple_tool_calls(self):
        """Model calls two tools before answering."""
        tool1 = make_anthropic_response(
            tool_name="get_azure_top_overages", tool_input={"n": 1},
        )
        tool2 = make_anthropic_response(
            tool_name="get_gcp_top_projects", tool_input={"n": 1},
        )
        final = make_anthropic_response(
            text="Azure: sub-a, GCP: proj-alpha"
        )

        mock_create = MagicMock(side_effect=[tool1, tool2, final])
        with patch("agent.Provider.create", mock_create):
            from agent import run
            answer, messages = run("Compare both clouds")

        assert "sub-a" in answer
        assert "proj-alpha" in answer
        assert mock_create.call_count == 3

    def test_conversation_memory_persists(self):
        """Messages list passed in should be appended to, not replaced."""
        tool_resp = make_anthropic_response(
            tool_name="get_azure_top_overages", tool_input={"n": 1},
        )
        final_resp = make_anthropic_response(text="sub-a is top.")

        mock_create = MagicMock(side_effect=[tool_resp, final_resp])
        with patch("agent.Provider.create", mock_create):
            from agent import run
            history = [{"role": "assistant", "content": "Previous answer"}]
            answer, messages = run("Which is top?", messages=history)

        # Original message should still be first
        assert messages[0] == {"role": "assistant", "content": "Previous answer"}
        # New messages should be appended
        assert len(messages) == 5  # history(1) + user(1) + assistant_toolcall + user_toolresult + assistant
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "sub-a is top."

    def test_history_pruning(self):
        """Very long history should be pruned to MAX_HISTORY."""
        tool_resp = make_anthropic_response(text="Done.")

        with patch("agent.Provider.create", return_value=tool_resp):
            from agent import run
            long_history = [
                {"role": "user", "content": f"Question {i}"}
                for i in range(30)
            ]
            answer, messages = run("Final question", messages=long_history)

        assert len(messages) <= 22  # 20 history + user + assistant


class TestProviderAbstraction:
    """Tests that the Provider works with both API shapes."""

    def test_anthropic_stop_reason_end_turn(self):
        resp = make_anthropic_response(text="Hello")
        from agent import Provider
        assert Provider.stop_reason(resp) == "end_turn"

    def test_anthropic_stop_reason_tool_use(self):
        resp = make_anthropic_response(tool_name="get_azure_top_overages")
        from agent import Provider
        assert Provider.stop_reason(resp) == "tool_use"

    def test_openai_stop_reason_stop(self):
        resp = make_openai_response(text="Hello")
        from agent import Provider
        assert Provider.stop_reason(resp) == "end_turn"

    def test_openai_stop_reason_tool_calls(self):
        resp = make_openai_response(tool_name="get_azure_top_overages")
        from agent import Provider
        assert Provider.stop_reason(resp) == "tool_use"

    def test_anthropic_content_blocks_text(self):
        resp = make_anthropic_response(text="Some answer")
        from agent import Provider
        blocks = list(Provider.iter_content_blocks(resp))
        assert len(blocks) == 1
        assert blocks[0].type == "text"
        assert blocks[0].text == "Some answer"

    def test_anthropic_content_blocks_tool(self):
        resp = make_anthropic_response(
            tool_name="get_azure_top_overages", tool_input={"n": 2},
        )
        from agent import Provider
        blocks = list(Provider.iter_content_blocks(resp))
        assert len(blocks) == 1
        assert blocks[0].type == "tool_use"
        assert blocks[0].name == "get_azure_top_overages"
        assert blocks[0].input == {"n": 2}

    def test_openai_content_blocks_text(self):
        resp = make_openai_response(text="Some answer")
        from agent import Provider
        blocks = list(Provider.iter_content_blocks(resp))
        assert len(blocks) == 1
        assert blocks[0].type == "text"
        assert blocks[0].text == "Some answer"

    def test_openai_content_blocks_tool(self):
        resp = make_openai_response(
            tool_name="get_gcp_top_projects", tool_input={"n": 3},
        )
        from agent import Provider
        blocks = list(Provider.iter_content_blocks(resp))
        assert len(blocks) == 1
        assert blocks[0].type == "tool_use"
        assert blocks[0].name == "get_gcp_top_projects"
        assert blocks[0].input == {"n": 3}

    def test_openai_tool_has_id(self):
        resp = make_openai_response(
            tool_name="get_azure_top_overages",
            tool_input={"n": 1},
            tool_id="call_xyz789",
        )
        from agent import Provider
        blocks = list(Provider.iter_content_blocks(resp))
        assert blocks[0].id == "call_xyz789"


class TestStreaming:
    """Tests for the streaming path through the agent.

    The streaming path in agent.py calls Provider.create(stream=True)
    then Provider.stream_response().  We mock both so no real API is hit.
    """

    def test_stream_simple_answer(self):
        """Streaming with no tools should still return the answer."""
        from agent import Provider

        # Mock the stream_response to return a pre-built response
        fake_response = make_anthropic_response(text="The top is sub-a.")

        with patch.object(Provider, "stream_response") as mock_stream:
            mock_stream.return_value = (fake_response, ["The top is sub-a."], [])
            with patch.object(Provider, "create") as mock_create:
                mock_create.return_value = "stream_obj_placeholder"

                from agent import run
                answer, messages = run("Which is top?", stream=True)

        assert "sub-a" in answer

    def test_stream_with_tool_call(self):
        """Streaming with a tool call should dispatch and continue.

        In streaming mode every turn calls backend.create(stream=True),
        so we need stream_response to work for all turns.  The mock
        keeps returning a tool-use on first call, then we switch to
        returning an end_turn for subsequent calls.
        """
        from agent import Provider

        tool_resp = make_anthropic_response(
            tool_name="get_azure_top_overages", tool_input={"n": 1},
        )
        final_resp = make_anthropic_response(text="Top is sub-a at $5000.")

        # stream_response is called once per streaming turn
        call_count = [0]

        def stream_response_side_effect(provider_name, stream_obj, *, verbose):
            call_count[0] += 1
            if call_count[0] == 1:
                # First turn: tool call
                return (tool_resp, [], [tool_resp.content[0]])
            else:
                # Second turn: end_turn
                return (final_resp, ["Top is sub-a at $5000."], [])

        with patch.object(Provider, "stream_response", side_effect=stream_response_side_effect):
            with patch.object(Provider, "create", return_value="stream_obj"):
                from agent import run
                answer, messages = run("Which is top?", stream=True)

        assert "sub-a" in answer
        assert "5000" in answer

    def test_stream_openai_provider(self):
        """Streaming path should work with OpenAI-compatible responses."""
        from agent import Provider

        fake_response = make_openai_response(text="Top is sub-a.")

        with patch.object(Provider, "stream_response") as mock_stream:
            mock_stream.return_value = (fake_response, ["Top is sub-a."], [])
            with patch.object(Provider, "create") as mock_create:
                mock_create.return_value = "stream_obj"

                from agent import run
                answer, messages = run(
                    "Which is top?", provider="deepseek", stream=True,
                )

        assert "sub-a" in answer


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_question(self):
        """Empty questions should still work."""
        fake_resp = make_anthropic_response(text="")

        with patch("agent.Provider.create", return_value=fake_resp):
            from agent import run
            answer, messages = run("")
        assert answer == ""

    def test_very_long_question(self):
        """Very long questions shouldn't break the loop."""
        long_q = "Which " + ("very " * 200) + "is top?"
        fake_resp = make_anthropic_response(text="sub-a")

        with patch("agent.Provider.create", return_value=fake_resp):
            from agent import run
            answer, messages = run(long_q)
        assert answer == "sub-a"

    def test_no_tool_definition_called(self):
        """Model ignores tools and just answers."""
        fake_resp = make_anthropic_response(text="Just answering directly.")

        with patch("agent.Provider.create", return_value=fake_resp):
            from agent import run
            answer, messages = run("Hello")
        assert "Just answering" in answer

    def test_build_assistant_content_anthropic(self):
        """_build_assistant_content should handle Anthropic responses."""
        from agent import _build_assistant_content
        resp = make_anthropic_response(
            tool_name="get_azure_top_overages", tool_input={"n": 1},
        )
        content = _build_assistant_content("anthropic", resp)
        assert len(content) == 1
        assert content[0]["type"] == "tool_use"
        assert content[0]["name"] == "get_azure_top_overages"
        assert content[0]["input"] == {"n": 1}

    def test_build_assistant_content_openai(self):
        """_build_assistant_content should handle OpenAI responses."""
        from agent import _build_assistant_content
        resp = make_openai_response(
            tool_name="get_azure_top_overages", tool_input={"n": 1},
        )
        content = _build_assistant_content("deepseek", resp)
        assert len(content) == 1
        assert content[0]["type"] == "tool_use"
        assert content[0]["name"] == "get_azure_top_overages"

    def test_build_assistant_content_with_override(self):
        """tool_blocks_override should be used when provided."""
        from agent import _build_assistant_content, _OaiToolCall

        raw = SimpleNamespace()
        raw.function = SimpleNamespace()
        raw.function.name = "compare_cross_cloud"
        raw.function.arguments = "{}"
        raw.id = "call_override"

        blocks = [_OaiToolCall(raw)]
        content = _build_assistant_content("deepseek", None, tool_blocks_override=blocks)
        assert len(content) == 1
        assert content[0]["name"] == "compare_cross_cloud"
        assert content[0]["id"] == "call_override"


class TestCLIParsing:
    """Tests for CLI argument parsing."""

    def test_defaults(self):
        from agent import _parse_args
        args = _parse_args([])
        assert args.provider == "anthropic"
        assert args.model is None
        assert args.verbose is False
        assert args.stream is False
        assert args.question is None

    def test_question_flag(self):
        from agent import _parse_args
        args = _parse_args(["-q", "Which is top?"])
        assert args.question == "Which is top?"

    def test_long_question_flag(self):
        from agent import _parse_args
        args = _parse_args(["--question", "Compare costs"])
        assert args.question == "Compare costs"

    def test_provider_choice(self):
        from agent import _parse_args
        args = _parse_args(["--provider", "deepseek"])
        assert args.provider == "deepseek"

    def test_model_flag(self):
        from agent import _parse_args
        args = _parse_args(["--model", "deepseek-chat"])
        assert args.model == "deepseek-chat"

    def test_verbose(self):
        from agent import _parse_args
        args = _parse_args(["-v"])
        assert args.verbose is True

    def test_stream(self):
        from agent import _parse_args
        args = _parse_args(["-s"])
        assert args.stream is True

    def test_all_flags(self):
        from agent import _parse_args
        args = _parse_args([
            "-q", "test", "--provider", "deepseek",
            "--model", "deepseek-chat", "-v", "-s",
        ])
        assert args.question == "test"
        assert args.provider == "deepseek"
        assert args.model == "deepseek-chat"
        assert args.verbose is True
        assert args.stream is True
