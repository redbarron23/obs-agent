#!/usr/bin/env python3
"""Multi-Cloud Cost Triage Agent — agent loop with tool-use.

The loop, the turn bookkeeping and the CLI live here; the provider layer
(`Provider`, message formatting, the system prompt, model config) is in
providers.py and re-exported below, so `from agent import Provider` still works.

Supports multiple LLM providers via a --provider flag (ollama [default], deepseek,
anthropic, or openai).

Usage
-----
Interactive REPL:
    uv run agent.py

Single question (scriptable):
    uv run agent.py -q "Which Azure subscription has the highest overage?"

Use Anthropic instead of the default (Ollama):
    export ANTHROPIC_API_KEY=sk-...
    uv run agent.py --provider anthropic -q "Show me the top 3 GCP projects"

Pick a specific Ollama model (no API key needed):
    uv run agent.py --model qwen2.5:7b -q "Compare Azure and GCP"

Use a specific model:
    uv run agent.py -q "Any cost spikes?" --model claude-sonnet-4-6

Run verbosely (see tool calls):
    uv run agent.py -q "Any cost spikes?" --verbose
"""

import argparse
import json
import sys
from collections.abc import Callable, Iterator

import observability
from providers import (  # noqa: F401  (re-exported for callers and tests)
    ANTHROPIC_ADAPTIVE_MODELS,
    ANTHROPIC_ADAPTIVE_PARAMS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    PROVIDER_DEFAULT_MODELS,
    PROVIDERS,
    SYSTEM,
    ConfigError,
    Provider,
    _format_messages_for_anthropic,
    _format_messages_for_openai,
    _OaiTextBlock,
    _OaiToolCall,
    _to_openai_tools,
    check_config,
    default_model,
    default_provider,
    max_tokens_setting,
    model_warning,
    served_by,
)
from tools import TOOL_DISPATCH, optional_tool_dispatch

MAX_TURNS = 10
MAX_HISTORY = 20


# ── Message formatting (canonical ↔ provider-specific) ─────────────────


def _prune_history(messages: list[dict], max_history: int = MAX_HISTORY) -> None:
    """Keep a protocol-valid suffix beginning with a user message."""
    if len(messages) <= max_history:
        return
    start = len(messages) - max_history
    while start < len(messages) and messages[start]["role"] != "user":
        start += 1
    messages[:] = messages[start:]


def _execute_tool(block, *, verbose: bool) -> str:
    """Dispatch a tool call (traced as a TOOL span when MLflow is on)."""
    with observability.span(block.name, "TOOL", inputs=dict(block.input or {})) as s:
        result = _dispatch_tool(block, verbose=verbose)
        observability.set_outputs(s, result)
        return result


def _dispatch_tool(block, *, verbose: bool) -> str:
    """Dispatch a tool call, returning an error string on failure."""
    name = block.name
    # Optional tools (search_observability_docs) join only when configured.
    dispatch = {**TOOL_DISPATCH, **optional_tool_dispatch()}
    if name not in dispatch:
        available = ", ".join(sorted(dispatch))
        msg = f"Error: unknown tool '{name}'. Available tools: {available}"
        if verbose:
            print(f"  [tool error: {msg}]", file=sys.stderr, flush=True)
        return msg
    try:
        return dispatch[name](block.input)
    except Exception as exc:
        msg = f"Error executing tool '{name}': {exc}"
        if verbose:
            print(f"  [tool error: {msg}]", file=sys.stderr, flush=True)
        return msg


# ── Provider abstraction ───────────────────────────────────────────────

# Claude models with always-on adaptive thinking and preserved thinking.


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
            for b in content_blocks
            if b.get("type") == "tool_use"
        ]
        text_parts = [b["text"] for b in content_blocks if b.get("type") == "text"]
        msg: dict = {"role": "assistant", "content": "\n".join(text_parts)}
        if tool_calls:
            msg = {
                "role": "assistant",
                "content": "\n".join(text_parts) if text_parts else None,
                "tool_calls": tool_calls,
            }
        raw = _raw_anthropic_content(response)
        if raw is not None:
            msg["anthropic_content"] = raw
        model = getattr(response, "model", None)
        if isinstance(model, str):
            msg["model"] = model
        return msg

    if tool_blocks_override is not None:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": tb.id, "name": tb.name, "input": tb.input} for tb in tool_blocks_override
            ],
        }

    tool_calls = []
    text_parts = []
    for choice in response.choices:
        if choice.message.content:
            text_parts.append(choice.message.content)
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": json.loads(tc.function.arguments),
                    }
                )
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
                file=sys.stderr,
                flush=True,
            )
        result = _execute_tool(block, verbose=verbose)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": block.id,
                "content": result,
            }
        )


def run(
    question: str,
    *,
    provider: str | None = None,
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
    provider = provider or default_provider()
    if model is None:
        model = default_model(provider)

    observability.setup()
    with observability.span(
        "agent",
        "AGENT",
        inputs={"question": question},
        attributes={"provider": provider, "model": model},
    ) as s:
        answer, messages = _run_loop(
            question,
            provider=provider,
            model=model,
            verbose=verbose,
            stream=stream,
            messages=messages,
            on_token=on_token,
        )
        observability.set_outputs(s, {"answer": answer, "served_by": served_by(messages)})
        return answer, messages


def _llm_span(provider: str, model: str, messages: list[dict]):
    return observability.span(
        f"llm: {provider}/{model}",
        "CHAT_MODEL",
        inputs={"messages": len(messages), "last": messages[-1].get("content")},
        attributes={"provider": provider, "model": model},
    )


def _llm_outputs(s, response, stop_reason: str, tool_calls: list[str]) -> None:
    usage = getattr(response, "usage", None)
    observability.set_outputs(
        s,
        {
            "stop_reason": stop_reason,
            "tool_calls": tool_calls,
            "served_by": getattr(response, "model", None)
            if isinstance(getattr(response, "model", None), str)
            else None,
            "input_tokens": getattr(usage, "input_tokens", None)
            or getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None)
            or getattr(usage, "completion_tokens", None),
        },
    )


def _run_loop(
    question: str,
    *,
    provider: str,
    model: str,
    verbose: bool,
    stream: bool,
    messages: list[dict] | None,
    on_token: Callable[[str], None] | None,
) -> tuple[str, list[dict]]:
    backend = Provider(provider, model)

    if messages is None:
        messages = []
    messages.append({"role": "user", "content": question})
    _prune_history(messages)

    for _turn in range(MAX_TURNS):
        if stream:
            with _llm_span(provider, model, messages) as s:
                stream_obj = backend.create(messages, stream=True)
                final_resp, text_blocks, tool_blocks = Provider.stream_response(
                    provider,
                    stream_obj,
                    verbose=verbose,
                    on_token=on_token,
                )
                sr = Provider.stop_reason(final_resp)
                _llm_outputs(s, final_resp, sr, [b.name for b in tool_blocks])

            if sr == "refusal":
                return _finish_refusal(final_resp, provider, messages, on_token)

            if sr == "end_turn":
                text_answer = "\n".join(text_blocks)
                messages.append(_final_message(provider, final_resp, text_answer))
                return text_answer, messages

            if sr == "tool_use":
                assistant_msg = _parse_assistant_response(
                    provider,
                    final_resp,
                    tool_blocks_override=tool_blocks,
                )
                messages.append(assistant_msg)
                _append_tool_results(messages, tool_blocks, verbose=verbose)
                continue

            text_answer = "\n".join(text_blocks)
            if sr == "max_tokens":
                return _finish_truncated(final_resp, provider, messages, on_token, text_answer)
            messages.append({"role": "assistant", "content": text_answer})
            return text_answer, messages

        with _llm_span(provider, model, messages) as s:
            response = backend.create(messages, stream=False)
            sr = Provider.stop_reason(response)
            _llm_outputs(
                s,
                response,
                sr,
                [
                    b.name
                    for b in Provider.iter_content_blocks(response)
                    if getattr(b, "type", None) == "tool_use"
                ],
            )
        if verbose:
            print(f"  [stop_reason: {sr}]", file=sys.stderr)

        if sr == "refusal":
            return _finish_refusal(response, provider, messages, None)

        text_blocks = [
            b.text
            for b in Provider.iter_content_blocks(response)
            if getattr(b, "type", "text") == "text" and hasattr(b, "text")
        ]

        if sr == "end_turn":
            text_answer = "\n".join(text_blocks)
            messages.append(_final_message(provider, response, text_answer))
            return text_answer, messages

        if sr == "tool_use":
            assistant_msg = _parse_assistant_response(provider, response)
            messages.append(assistant_msg)

            tool_blocks = [
                b for b in Provider.iter_content_blocks(response) if b.type == "tool_use"
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
        if sr == "max_tokens":
            return _finish_truncated(response, provider, messages, None, text_answer)
        messages.append({"role": "assistant", "content": text_answer})
        return text_answer, messages

    return (
        f"Agent stopped after {MAX_TURNS} tool-use turns without a final answer.",
        messages,
    )


def stream_run(
    question: str,
    *,
    provider: str | None = None,
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
        try:
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
        except Exception as exc:  # surface to the consuming generator
            result["error"] = exc
        finally:
            token_queue.put(None)

    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()
    while True:
        token = token_queue.get()
        if token is None:
            break
        yield token
    thread.join()

    if "error" in result:
        raise result["error"]
    if result_holder is not None:
        result_holder.update(result)


def _final_message(provider: str, response, text: str) -> dict:
    """Canonical final-answer message; Anthropic keeps its raw blocks for replay."""
    if provider == "anthropic":
        msg = _parse_assistant_response(provider, response)
        msg["content"] = text
        return msg
    return {"role": "assistant", "content": text}


def _finish_truncated(response, provider: str, messages: list[dict], on_token, partial: str):
    """End the turn when the model hit the token cap, saying so explicitly.

    Returning the partial text as if it were a complete answer is the worst
    outcome: the caller cannot tell a short answer from a cut-off one. Instead
    keep whatever streamed, then append a note naming the setting to raise.
    """
    note = (
        "\n\n[Response truncated: the model hit the output-token limit before finishing. "
        "Raise OBS_AGENT_MAX_TOKENS (currently "
        f"{max_tokens_setting()}) or ask a narrower question.]"
    )
    text = (partial + note) if partial else note.lstrip("\n")
    if on_token:
        on_token(note)
    messages.append(_final_message(provider, response, text))
    return text, messages


def _finish_refusal(response, provider: str, messages: list[dict], on_token):
    """End the turn on a safety refusal with an explanation instead of silence."""
    text = _refusal_message(response)
    if on_token:
        on_token(text)
    msg = _final_message(provider, response, text)
    messages.append(msg)
    return text, messages


def _raw_anthropic_content(response) -> list[dict] | None:
    """The response's content blocks as JSON dicts, unchanged, for replay.

    Only real SDK objects qualify; test doubles fall back to the rebuilt form.
    """
    blocks = getattr(response, "content", None) or []
    if not blocks or not all(hasattr(b, "model_dump") and callable(b.model_dump) for b in blocks):
        return None
    try:
        dumped = [b.model_dump(mode="json", exclude_none=True) for b in blocks]
    except Exception:
        return None
    return dumped if all(isinstance(d, dict) for d in dumped) else None


def _refusal_message(response) -> str:
    details = getattr(response, "stop_details", None)
    category = getattr(details, "category", None)
    suffix = f" (category: {category})" if isinstance(category, str) else ""
    return (
        f"The model declined to answer this request{suffix}. "
        "Try rephrasing, or use another provider."
    )


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
        "-q",
        "--question",
        help="Single question to answer (omit for interactive REPL).",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=list(PROVIDERS),
        help=f"LLM provider (default: $OBS_AGENT_PROVIDER, else {DEFAULT_PROVIDER}).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: $OBS_AGENT_MODEL for the default provider, "
        "else the provider's default).",
    )
    parser.add_argument(
        "--stream",
        "-s",
        action="store_true",
        help="Stream output tokens as they arrive (instead of printing all at once).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show tool calls and stop reasons.",
    )
    return parser.parse_args(argv)


def _repl(provider: str, model: str, verbose: bool, stream: bool) -> None:
    """Run the interactive REPL with conversation memory."""
    print("obs-agent — Multi-Cloud Cost Triage Agent")
    mode = "stream" if stream else "batch"
    print(f"Provider: {provider}  |  Model: {model}  |  Mode: {mode}  |  Type 'quit' to exit.\n")
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
    try:
        args.provider = args.provider or default_provider()
        model = args.model or default_model(args.provider)
    except ConfigError as exc:
        sys.exit(f"obs-agent: {exc}")
    warning = model_warning(args.provider, model)
    if warning and (args.verbose or not args.question):
        print(f"Note: {warning}", file=sys.stderr)

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
            answer, messages = run(
                args.question,
                provider=args.provider,
                model=model,
                verbose=args.verbose,
                stream=False,
            )
            print(answer)
            if args.verbose:
                print(f"  [answered by {served_by(messages) or model}]", file=sys.stderr)
    else:
        _repl(
            provider=args.provider,
            model=model,
            verbose=args.verbose,
            stream=args.stream,
        )


if __name__ == "__main__":
    main()
