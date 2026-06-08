#!/usr/bin/env python3
"""Multi-Cloud Cost Triage Agent — agent loop with tool-use.

Supports multiple LLM providers via a --provider flag (anthropic or deepseek).

Usage
-----
Interactive REPL:
    python agent.py

Single question (scriptable):
    python agent.py -q "Which Azure subscription has the highest overage?"

Use DeepSeek instead of Anthropic:
    export DEEPSEEK_API_KEY=sk-...
    python agent.py --provider deepseek -q "Show me the top 3 GCP projects"

Use a specific model:
    python agent.py -q "Any cost spikes?" --model claude-sonnet-4-6

Run verbosely (see tool calls):
    python agent.py -q "Any cost spikes?" --verbose
"""

import argparse
import os
import sys

from tools import TOOL_DEFINITIONS, TOOL_DISPATCH

# ── System prompt ──────────────────────────────────────────────────────

SYSTEM = """You are a multi-cloud cost triage assistant.
You have access to cloud logging billing data for Azure and GCP.
When answering questions, always call the relevant tools to get actual numbers.
Be concise and highlight the most important findings."""


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
        else:
            raise ValueError(f"Unknown provider '{name}'. Use 'anthropic' or 'deepseek'.")

    def create(self, messages: list[dict], *, stream: bool = False) -> object:
        """Send a message list and return the API response."""
        if self.name == "anthropic":
            kwargs = dict(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )
            if stream:
                return self._client.messages.stream(**kwargs)
            return self._client.messages.create(**kwargs)
        else:  # deepseek (OpenAI-compatible)
            oai_messages = [{"role": "system", "content": SYSTEM}] + messages
            kwargs = dict(
                model=self.model,
                max_tokens=1024,
                tools=TOOL_DEFINITIONS,
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
        else:  # OpenAI-compatible
            reason = response.choices[0].finish_reason or ""
            # Map OpenAI reasons to our expected values
            mapping = {
                "stop": "end_turn",
                "tool_calls": "tool_use",
            }
            return mapping.get(reason, reason)

    # ── Streaming helpers ──────────────────────────────────────────

    @staticmethod
    def stream_response(provider_name: str, stream_obj, *, verbose: bool):
        """
        Consume a streaming response, yielding tokens and returning the
        accumulated content blocks and final (non-streaming) response object
        (so we can read tool calls / stop reasons off it).

        Yields (token: str | None) for printed output.
        Returns (response_object, text_blocks_list, tool_use_blocks_list).
        """
        if provider_name == "anthropic":
            return Provider._stream_anthropic(stream_obj, verbose=verbose)
        else:
            return Provider._stream_openai(stream_obj, verbose=verbose)

    @staticmethod
    def _stream_anthropic(stream, *, verbose: bool):
        """Consume an Anthropic message stream."""
        collected = {"text_blocks": [], "tool_use_blocks": []}
        current_text = []

        with stream as s:
            for event in s:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True)
                    current_text.append(event.delta.text)
                elif event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        collected["tool_use_blocks"].append(event.content_block)
                        if verbose:
                            func_name = event.content_block.name
                            print(f"\n  [tool: {func_name}(...)]", file=sys.stderr, flush=True)
                elif event.type == "message_delta":
                    if event.delta.stop_reason == "tool_use" and verbose:
                        pass  # already logged above

                elif event.type == "content_block_stop":
                    if current_text:
                        collected["text_blocks"].append("".join(current_text))
                        current_text = []

                elif event.type == "message_start":
                    for block in event.message.content:
                        if block.type == "tool_use":
                            collected["tool_use_blocks"].append(block)
                            if verbose:
                                print(f"\n  [tool: {block.name}({block.input})]", file=sys.stderr, flush=True)

            # Flush remaining text
            if current_text:
                collected["text_blocks"].append("".join(current_text))

            final_response = s.get_final_message()
            text_str = "\n".join(collected["text_blocks"])
            return final_response, collected["text_blocks"], collected["tool_use_blocks"]

    @staticmethod
    def _stream_openai(stream, *, verbose: bool):
        """Consume an OpenAI-compatible (DeepSeek) stream."""
        from openai import Stream
        collected_text = []
        tool_call_chunks = {}  # index -> {id, function}

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # Text delta
            if delta.content:
                print(delta.content, end="", flush=True)
                collected_text.append(delta.content)

            # Tool call deltas (can arrive in multiple chunks per call)
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
                            tool_call_chunks[idx]["function"]["arguments"] += tc_chunk.function.arguments

            # Log tool name when first seen
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name and verbose:
                        print(f"\n  [tool: {tc.function.name}(...)]", file=sys.stderr, flush=True)

        print()  # newline after stream

        # Reconstruct the tool call blocks
        import json
        tool_use_blocks = []
        for idx in sorted(tool_call_chunks):
            tcc = tool_call_chunks[idx]
            tool_use_blocks.append(
                _OaiToolCall.from_dict({
                    "id": tcc["id"],
                    "function": {
                        "name": tcc["function"]["name"],
                        "arguments": tcc["function"]["arguments"],
                    }
                })
            )

        text_blocks = ["".join(collected_text)] if collected_text else []
        # Build a dummy response-like object for stop_reason etc.
        final_reason = chunk.choices[0].finish_reason if chunk.choices else "stop"

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
        import json
        self.input = json.loads(raw.function.arguments)
        self.id = raw.id

    @classmethod
    def from_dict(cls, d):
        """Build from a reconstructed dict (streaming path)."""
        obj = cls.__new__(cls)
        obj.type = "tool_use"
        obj.name = d["function"]["name"]
        import json
        obj.input = json.loads(d["function"]["arguments"])
        obj.id = d["id"]
        return obj


class _OaiTextBlock:
    """Lightweight stand-in for an Anthropic text block."""
    def __init__(self, text):
        self.type = "text"
        self.text = text


# ── Defaults ───────────────────────────────────────────────────────────

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-sonnet-4-6"

# Model defaults per provider (used when --model is not given)
PROVIDER_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "deepseek": "deepseek-chat",
}


# ── Agent loop ─────────────────────────────────────────────────────────

def run(
    question: str,
    *,
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    verbose: bool = False,
    stream: bool = False,
    messages: list[dict] | None = None,
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

    # Use provided history or start fresh
    if messages is None:
        messages = []
    messages.append({"role": "user", "content": question})

    # Trim history if it gets too long (simple strategy: keep last 20)
    # We keep the system prompt setup out of messages, so this is safe.
    MAX_HISTORY = 20
    if len(messages) > MAX_HISTORY:
        # Keep only the last N messages
        messages[:] = messages[-MAX_HISTORY:]

    while True:
        # ── Streaming turn ────────────────────────────────────────
        if stream:
            stream_obj = backend.create(messages, stream=True)
            final_resp, text_blocks, tool_blocks = Provider.stream_response(
                provider, stream_obj, verbose=verbose,
            )
            sr = Provider.stop_reason(final_resp)

            if sr == "end_turn":
                text_answer = "\n".join(text_blocks)
                messages.append({
                    "role": "assistant",
                    "content": text_answer,
                })
                return text_answer, messages

            if sr == "tool_use":
                assistant_content = _build_assistant_content(
                    provider, final_resp, tool_blocks_override=tool_blocks,
                )
                messages.append({
                    "role": "assistant",
                    "content": assistant_content,
                })

                tool_results = []
                for block in tool_blocks:
                    if verbose:
                        print(
                            f"  [tool result: {block.name}({block.input})]",
                            file=sys.stderr,
                            flush=True,
                        )
                    result = TOOL_DISPATCH[block.name](block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

                messages.append({"role": "user", "content": tool_results})
                continue

            text_answer = "\n".join(text_blocks)
            messages.append({
                "role": "assistant",
                "content": text_answer,
            })
            return text_answer, messages

        # ── Non-streaming turn ────────────────────────────────────
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
            messages.append({
                "role": "assistant",
                "content": text_answer,
            })
            return text_answer, messages

        if sr == "tool_use":
            assistant_content = _build_assistant_content(
                provider, response
            )
            messages.append({
                "role": "assistant",
                "content": assistant_content,
            })

            tool_results = []
            for block in Provider.iter_content_blocks(response):
                if block.type == "tool_use":
                    if verbose:
                        print(
                            f"  [tool: {block.name}({block.input})]",
                            file=sys.stderr,
                        )
                    result = TOOL_DISPATCH[block.name](block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})
        else:
            text_answer = "\n".join(text_blocks)
            messages.append({
                "role": "assistant",
                "content": text_answer,
            })
            return text_answer, messages


def _build_assistant_content(
    provider: str,
    response: object,
    tool_blocks_override: list | None = None,
) -> list:
    """Build the assistant content list from the response.

    When streaming we pass tool_blocks_override because the
    stream may have reconstructed the tool calls already.
    """
    if provider == "anthropic":
        # Convert SDK content blocks to plain dicts so subsequent API calls
        # can serialize the messages list correctly.
        return [_content_block_to_dict(b) for b in response.content]

    # OpenAI-compatible: construct tool call blocks
    content = []

    # If we have pre-built tool blocks from streaming, use those
    if tool_blocks_override is not None:
        for tb in tool_blocks_override:
            content.append({
                "type": "tool_use",
                "id": tb.id,
                "name": tb.name,
                "input": tb.input,
            })
        return content

    for choice in response.choices:
        if choice.message.content:
            content.append({
                "type": "text",
                "text": choice.message.content,
            })
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                import json
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                })
    return content


def _content_block_to_dict(block) -> dict:
    """Convert an Anthropic SDK content block to a plain dict.

    This is needed because the Anthropic SDK returns content blocks as
    SDK objects (TextBlock, ToolUseBlock), but we store them in the
    messages list.  If we pass SDK objects back on subsequent API calls,
    the SDK may not serialize them correctly, leading to API errors
    like "Cannot continue from message role: assistant".
    """
    if block.type == "text":
        return {"type": "text", "text": block.text}
    elif block.type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    else:
        return {"type": block.type}


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
        choices=["anthropic", "deepseek"],
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
    print(f"Provider: {provider}  |  Model: {model}  |  Mode: {mode}  |  Type 'quit' to exit.\n")
    messages: list[dict] = []  # persistent conversation history
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
            answer, _messages = run(
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
        _repl(provider=args.provider, model=model, verbose=args.verbose, stream=args.stream)


if __name__ == "__main__":
    main()
