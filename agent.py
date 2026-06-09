#!/usr/bin/env python3
"""Multi-Cloud Cost Triage Agent — agent loop with tool-use.

Supports multiple LLM providers via a --provider flag (deepseek, anthropic, or ollama).

Usage
-----
Interactive REPL:
    python agent.py

Single question (scriptable):
    python agent.py -q "Which Azure subscription has the highest overage?"

Use Anthropic instead of DeepSeek:
    export ANTHROPIC_API_KEY=sk-...
    python agent.py --provider anthropic -q "Show me the top 3 GCP projects"

Run Ollama locally (no API key needed):
    python agent.py --provider ollama --model llama3.2 -q "Compare Azure and GCP"

Use a specific model:
    python agent.py -q "Any cost spikes?" --model claude-sonnet-4-6

Run verbosely (see tool calls):
    python agent.py -q "Any cost spikes?" --verbose
"""

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterator

from tools import TOOL_DEFINITIONS, TOOL_DISPATCH

# ── System prompt ──────────────────────────────────────────────────────

SYSTEM = """You are a multi-cloud cost triage assistant.
You have access to cloud logging billing data for Azure and GCP.
When answering questions, always call the relevant tools to get actual numbers.
Be concise and highlight the most important findings."""

# ── Limits ─────────────────────────────────────────────────────────────

MAX_TURNS = 10
MAX_HISTORY = 20


# ── Message formatting (canonical ↔ provider-specific) ─────────────────

def _format_messages_for_anthropic(messages: list[dict]) -> list[dict]:
    """Convert canonical internal messages to Anthropic API format."""
    formatted: list[dict] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg["role"] == "tool":
            tool_results = []
            while i < len(messages) and messages[i]["role"] == "tool":
                t = messages[i]
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": t["tool_call_id"],
                    "content": t["content"],
                })
                i += 1
            formatted.append({"role": "user", "content": tool_results})
            continue
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            content = []
            if msg.get("content"):
                content.append({"type": "text", "text": msg["content"]})
            for tc in msg["tool_calls"]:
                content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"],
                })
            formatted.append({"role": "assistant", "content": content})
        else:
            formatted.append(msg)
        i += 1
    return formatted


def _format_messages_for_openai(messages: list[dict]) -> list[dict]:
    """Convert canonical internal messages to OpenAI API format."""
    formatted: list[dict] = []
    for msg in messages:
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            oai_msg: dict = {
                "role": "assistant",
                "content": msg.get("content"),
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["input"]),
                        },
                    }
                    for tc in msg["tool_calls"]
                ],
            }
            formatted.append(oai_msg)
        elif msg["role"] == "tool":
            formatted.append({
                "role": "tool",
                "tool_call_id": msg["tool_call_id"],
                "content": msg["content"],
            })
        else:
            formatted.append(msg)
    return formatted


def _to_openai_tools(tool_definitions: list[dict]) -> list[dict]:
    """Convert Anthropic-style tool defs to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tool_definitions
    ]


def _prune_history(messages: list[dict], max_history: int = MAX_HISTORY) -> None:
    """Keep the first user message plus the most recent turns."""
    if len(messages) <= max_history:
        return
    first = messages[0]
    messages[:] = [first] + messages[-(max_history - 1):]


def _execute_tool(block, *, verbose: bool) -> str:
    """Dispatch a tool call, returning an error string on failure."""
    name = block.name
    if name not in TOOL_DISPATCH:
        available = ", ".join(sorted(TOOL_DISPATCH))
        msg = f"Error: unknown tool '{name}'. Available tools: {available}"
        if verbose:
            print(f"  [tool error: {msg}]", file=sys.stderr, flush=True)
        return msg
    try:
        return TOOL_DISPATCH[name](block.input)
    except Exception as exc:
        msg = f"Error executing tool '{name}': {exc}"
        if verbose:
            print(f"  [tool error: {msg}]", file=sys.stderr, flush=True)
        return msg


def _emit_token(token: str, on_token: Callable[[str], None] | None) -> None:
    """Send a token to a callback or stdout."""
    if on_token:
        on_token(token)
    else:
        print(token, end="", flush=True)


# ── Provider abstraction ───────────────────────────────────────────────

class Provider:
    """Minimal wrapper over Anthropic and OpenAI-compatible APIs."""

    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model

        if name == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic()
        elif name == "deepseek":
            from openai import OpenAI
            self._client = OpenAI(
                base_url="https://api.deepseek.com",
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
            )
        elif name == "ollama":
            from openai import OpenAI
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            self._client = OpenAI(
                base_url=f"{host}/v1",
                api_key="ollama",  # no real auth needed
            )
        else:
            raise ValueError(
                f"Unknown provider '{name}'. Use 'anthropic', 'deepseek', or 'ollama'."
            )

    def create(self, messages: list[dict], *, stream: bool = False) -> object:
        """Send a message list and return the API response."""
        if self.name == "anthropic":
            api_messages = _format_messages_for_anthropic(messages)
            kwargs = dict(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM,
                tools=TOOL_DEFINITIONS,
                messages=api_messages,
            )
            if stream:
                return self._client.messages.stream(**kwargs)
            return self._client.messages.create(**kwargs)

        api_messages = _format_messages_for_openai(messages)
        oai_messages = [{"role": "system", "content": SYSTEM}] + api_messages
        kwargs = dict(
            model=self.model,
            max_tokens=1024,
            tools=_to_openai_tools(TOOL_DEFINITIONS),
            messages=oai_messages,
        )
        if stream:
            return self._client.chat.completions.create(**kwargs, stream=True)
        return self._client.chat.completions.create(**kwargs)

    @staticmethod
    def iter_content_blocks(response: object):
        """Yield content blocks from the response (provider-agnostic)."""
        if hasattr(response, "content"):  # Anthropic
            yield from response.content
        else:  # OpenAI-compatible
            for choice in response.choices:
                if choice.message.tool_calls:
                    for tc in choice.message.tool_calls:
                        yield _OaiToolCall(tc)
                elif choice.message.content:
                    yield _OaiTextBlock(choice.message.content)

    @staticmethod
    def stop_reason(response: object) -> str:
        """Return a normalised stop reason."""
        if hasattr(response, "stop_reason"):  # Anthropic
            return response.stop_reason or ""
        reason = response.choices[0].finish_reason or ""
        mapping = {"stop": "end_turn", "tool_calls": "tool_use"}
        return mapping.get(reason, reason)

    @staticmethod
    def stream_response(
        provider_name: str,
        stream_obj,
        *,
        verbose: bool,
        on_token: Callable[[str], None] | None = None,
    ):
        """Consume a streaming response and return accumulated blocks."""
        if provider_name == "anthropic":
            return Provider._stream_anthropic(
                stream_obj, verbose=verbose, on_token=on_token,
            )
        return Provider._stream_openai(
            stream_obj, verbose=verbose, on_token=on_token,
        )

    @staticmethod
    def _stream_anthropic(stream, *, verbose: bool, on_token):
        """Consume an Anthropic message stream."""
        collected = {"text_blocks": [], "tool_use_blocks": []}
        current_text = []

        with stream as s:
            for event in s:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    _emit_token(event.delta.text, on_token)
                    current_text.append(event.delta.text)
                elif event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        collected["tool_use_blocks"].append(event.content_block)
                        if verbose:
                            func_name = event.content_block.name
                            print(
                                f"\n  [tool: {func_name}(...)]",
                                file=sys.stderr, flush=True,
                            )
                elif event.type == "content_block_stop":
                    if current_text:
                        collected["text_blocks"].append("".join(current_text))
                        current_text = []
                elif event.type == "message_start":
                    for block in event.message.content:
                        if block.type == "tool_use":
                            collected["tool_use_blocks"].append(block)
                            if verbose:
                                print(
                                    f"\n  [tool: {block.name}({block.input})]",
                                    file=sys.stderr, flush=True,
                                )

            if current_text:
                collected["text_blocks"].append("".join(current_text))

            final_response = s.get_final_message()

        if on_token is None:
            print()
        return final_response, collected["text_blocks"], collected["tool_use_blocks"]

    @staticmethod
    def _stream_openai(stream, *, verbose: bool, on_token):
        """Consume an OpenAI-compatible (DeepSeek/Ollama) stream."""
        collected_text = []
        tool_call_chunks = {}
        chunk = None

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            if delta.content:
                _emit_token(delta.content, on_token)
                collected_text.append(delta.content)

            if delta.tool_calls:
                for tc_chunk in delta.tool_calls:
                    idx = tc_chunk.index
                    if idx not in tool_call_chunks:
                        tool_call_chunks[idx] = {
                            "id": tc_chunk.id or "",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc_chunk.id:
                        tool_call_chunks[idx]["id"] = tc_chunk.id
                    if tc_chunk.function:
                        if tc_chunk.function.name:
                            tool_call_chunks[idx]["function"]["name"] = tc_chunk.function.name
                        if tc_chunk.function.arguments:
                            tool_call_chunks[idx]["function"]["arguments"] += (
                                tc_chunk.function.arguments
                            )
                    if tc_chunk.function and tc_chunk.function.name and verbose:
                        print(
                            f"\n  [tool: {tc_chunk.function.name}(...)]",
                            file=sys.stderr, flush=True,
                        )

        if on_token is None:
            print()

        tool_use_blocks = [
            _OaiToolCall.from_dict({
                "id": tcc["id"],
                "function": {
                    "name": tcc["function"]["name"],
                    "arguments": tcc["function"]["arguments"],
                },
            })
            for _, tcc in sorted(tool_call_chunks.items())
        ]

        text_blocks = ["".join(collected_text)] if collected_text else []
        final_reason = chunk.choices[0].finish_reason if chunk and chunk.choices else "stop"

        class _DummyResponse:
            def __init__(self, reason, blocks):
                self.stop_reason = reason
                self.content = blocks
                self.choices = [type("Choice", (), {"finish_reason": reason})()]

        return _DummyResponse(final_reason, tool_use_blocks), text_blocks, tool_use_blocks


class _OaiToolCall:
    """Lightweight stand-in for an Anthropic tool-use block."""

    def __init__(self, raw):
        self.type = "tool_use"
        self.name = raw.function.name
        self.input = json.loads(raw.function.arguments)
        self.id = raw.id

    @classmethod
    def from_dict(cls, d):
        """Build from a reconstructed dict (streaming path)."""
        obj = cls.__new__(cls)
        obj.type = "tool_use"
        obj.name = d["function"]["name"]
        obj.input = json.loads(d["function"]["arguments"])
        obj.id = d["id"]
        return obj


class _OaiTextBlock:
    """Lightweight stand-in for an Anthropic text block."""

    def __init__(self, text):
        self.type = "text"
        self.text = text


# ── Defaults ───────────────────────────────────────────────────────────

DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-chat"

PROVIDER_DEFAULT_MODELS = {
    "deepseek": "deepseek-chat",
    "anthropic": "claude-sonnet-4-6",
    "ollama": "llama3.2",
}


# ── Agent loop ─────────────────────────────────────────────────────────

def _parse_assistant_response(
    provider: str,
    response: object,
    tool_blocks_override: list | None = None,
) -> dict:
    """Return a canonical assistant message dict from an API response."""
    if provider == "anthropic":
        content_blocks = [_content_block_to_dict(b) for b in response.content]
        tool_calls = [
            {"id": b["id"], "name": b["name"], "input": b["input"]}
            for b in content_blocks if b.get("type") == "tool_use"
        ]
        text_parts = [b["text"] for b in content_blocks if b.get("type") == "text"]
        if tool_calls:
            return {
                "role": "assistant",
                "content": "\n".join(text_parts) if text_parts else None,
                "tool_calls": tool_calls,
            }
        return {"role": "assistant", "content": "\n".join(text_parts)}

    if tool_blocks_override is not None:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": tb.id, "name": tb.name, "input": tb.input}
                for tb in tool_blocks_override
            ],
        }

    tool_calls = []
    text_parts = []
    for choice in response.choices:
        if choice.message.content:
            text_parts.append(choice.message.content)
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                })
    if tool_calls:
        return {
            "role": "assistant",
            "content": "\n".join(text_parts) if text_parts else None,
            "tool_calls": tool_calls,
        }
    return {"role": "assistant", "content": "\n".join(text_parts)}


def _append_tool_results(
    messages: list[dict],
    tool_blocks: list,
    *,
    verbose: bool,
) -> None:
    """Execute tools and append canonical tool-result messages."""
    for block in tool_blocks:
        if verbose:
            print(
                f"  [tool result: {block.name}({block.input})]",
                file=sys.stderr, flush=True,
            )
        result = _execute_tool(block, verbose=verbose)
        messages.append({
            "role": "tool",
            "tool_call_id": block.id,
            "content": result,
        })


def run(
    question: str,
    *,
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    verbose: bool = False,
    stream: bool = False,
    messages: list[dict] | None = None,
    on_token: Callable[[str], None] | None = None,
) -> tuple[str, list[dict]]:
    """Run a single question through the agent loop.

    Parameters
    ----------
    question : str
        The user's question.
    provider, model, verbose, stream :
        Standard options (see --help).
    messages : list[dict] | None
        Optional conversation history to continue from. If None,
        starts a new conversation. The list is mutated in place.
    on_token : callable | None
        Optional callback invoked with each streamed text token.

    Returns
    -------
    (answer_text, updated_messages)
        The text answer and the full message history (including tool
        calls and results), which can be passed back on subsequent calls
        for multi-turn conversation.
    """
    if model is None:
        model = PROVIDER_DEFAULT_MODELS.get(provider, DEFAULT_MODEL)

    backend = Provider(provider, model)

    if messages is None:
        messages = []
    messages.append({"role": "user", "content": question})
    _prune_history(messages)

    for _turn in range(MAX_TURNS):
        if stream:
            stream_obj = backend.create(messages, stream=True)
            final_resp, text_blocks, tool_blocks = Provider.stream_response(
                provider, stream_obj, verbose=verbose, on_token=on_token,
            )
            sr = Provider.stop_reason(final_resp)

            if sr == "end_turn":
                text_answer = "\n".join(text_blocks)
                messages.append({"role": "assistant", "content": text_answer})
                return text_answer, messages

            if sr == "tool_use":
                assistant_msg = _parse_assistant_response(
                    provider, final_resp, tool_blocks_override=tool_blocks,
                )
                messages.append(assistant_msg)
                _append_tool_results(messages, tool_blocks, verbose=verbose)
                continue

            text_answer = "\n".join(text_blocks)
            messages.append({"role": "assistant", "content": text_answer})
            return text_answer, messages

        response = backend.create(messages, stream=False)
        sr = Provider.stop_reason(response)
        if verbose:
            print(f"  [stop_reason: {sr}]", file=sys.stderr)

        text_blocks = [
            b.text for b in Provider.iter_content_blocks(response)
            if hasattr(b, "text")
        ]

        if sr == "end_turn":
            text_answer = "\n".join(text_blocks)
            messages.append({"role": "assistant", "content": text_answer})
            return text_answer, messages

        if sr == "tool_use":
            assistant_msg = _parse_assistant_response(provider, response)
            messages.append(assistant_msg)

            tool_blocks = [
                b for b in Provider.iter_content_blocks(response)
                if b.type == "tool_use"
            ]
            for block in tool_blocks:
                if verbose:
                    print(
                        f"  [tool: {block.name}({block.input})]",
                        file=sys.stderr,
                    )
            _append_tool_results(messages, tool_blocks, verbose=verbose)
            continue

        text_answer = "\n".join(text_blocks)
        messages.append({"role": "assistant", "content": text_answer})
        return text_answer, messages

    return (
        f"Agent stopped after {MAX_TURNS} tool-use turns without a final answer.",
        messages,
    )


def stream_run(
    question: str,
    *,
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    verbose: bool = False,
    messages: list[dict] | None = None,
    result_holder: dict | None = None,
) -> Iterator[str]:
    """Yield text tokens as the agent generates them.

    If *result_holder* is provided, it is populated with ``answer`` and
    ``messages`` keys when streaming completes.
    """
    import queue
    import threading

    token_queue: queue.SimpleQueue[str | None] = queue.SimpleQueue()
    result: dict = {}

    def on_token(token: str) -> None:
        token_queue.put(token)

    def run_agent() -> None:
        answer, history = run(
            question,
            provider=provider,
            model=model,
            verbose=verbose,
            stream=True,
            messages=messages,
            on_token=on_token,
        )
        result["answer"] = answer
        result["messages"] = history
        token_queue.put(None)

    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()
    while True:
        token = token_queue.get()
        if token is None:
            break
        yield token
    thread.join()

    if result_holder is not None:
        result_holder.update(result)


def _content_block_to_dict(block) -> dict:
    """Convert an Anthropic SDK content block to a plain dict."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    return {"type": block.type}


# Backward-compatible alias used by tests
_build_assistant_content = _parse_assistant_response


# ── CLI ────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Cloud Cost Triage Agent",
    )
    parser.add_argument(
        "-q", "--question",
        help="Single question to answer (omit for interactive REPL).",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["anthropic", "deepseek", "ollama"],
        help=f"LLM provider (default: {DEFAULT_PROVIDER}).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (provider-specific default if omitted).",
    )
    parser.add_argument(
        "--stream", "-s",
        action="store_true",
        help="Stream output tokens as they arrive (instead of printing all at once).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show tool calls and stop reasons.",
    )
    return parser.parse_args(argv)


def _repl(provider: str, model: str, verbose: bool, stream: bool) -> None:
    """Run the interactive REPL with conversation memory."""
    print("obs-agent — Multi-Cloud Cost Triage Agent")
    mode = "stream" if stream else "batch"
    print(
        f"Provider: {provider}  |  Model: {model}  |  Mode: {mode}  "
        f"|  Type 'quit' to exit.\n"
    )
    messages: list[dict] = []
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        if stream:
            print("Agent: ", end="", flush=True)
            answer, messages = run(
                question,
                provider=provider,
                model=model,
                verbose=verbose,
                stream=True,
                messages=messages,
            )
            print()
        else:
            answer, messages = run(
                question,
                provider=provider,
                model=model,
                verbose=verbose,
                stream=False,
                messages=messages,
            )
            print(f"\nAgent: {answer}\n")


def main() -> None:
    args = _parse_args()
    model = args.model or PROVIDER_DEFAULT_MODELS.get(args.provider, DEFAULT_MODEL)

    if args.question:
        if args.stream:
            run(
                args.question,
                provider=args.provider,
                model=model,
                verbose=args.verbose,
                stream=True,
            )
        else:
            answer, _messages = run(
                args.question,
                provider=args.provider,
                model=model,
                verbose=args.verbose,
                stream=False,
            )
            print(answer)
    else:
        _repl(
            provider=args.provider,
            model=model,
            verbose=args.verbose,
            stream=args.stream,
        )


if __name__ == "__main__":
    main()
