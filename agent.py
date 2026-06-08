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

    def create(self, messages: list[dict]) -> object:
        """Send a message list and return the API response."""
        if self.name == "anthropic":
            return self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )
        else:  # deepseek (OpenAI-compatible)
            # OpenAI doesn't support a separate system parameter in messages.create
            # — prepend system prompt as a system message instead.
            oai_messages = [{"role": "system", "content": SYSTEM}] + messages
            return self._client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                tools=TOOL_DEFINITIONS,
                messages=oai_messages,
            )

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


class _OaiToolCall:
    """Lightweight stand-in for an Anthropic tool-use block."""
    def __init__(self, raw):
        self.type = "tool_use"
        self.name = raw.function.name
        import json
        self.input = json.loads(raw.function.arguments)
        self.id = raw.id


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
) -> str:
    """Run a single question through the agent loop and return the answer."""
    if model is None:
        model = PROVIDER_DEFAULT_MODELS.get(provider, DEFAULT_MODEL)

    backend = Provider(provider, model)
    messages: list[dict] = [{"role": "user", "content": question}]

    while True:
        response = backend.create(messages)

        sr = Provider.stop_reason(response)
        if verbose:
            print(f"  [stop_reason: {sr}]", file=sys.stderr)

        # Collect text blocks from the response
        text_blocks = [
            b.text for b in Provider.iter_content_blocks(response)
            if hasattr(b, "text")
        ]

        if sr == "end_turn":
            return "\n".join(text_blocks)

        if sr == "tool_use":
            # Append assistant message with tool-use blocks
            assistant_content = _build_assistant_content(
                provider, response
            )
            messages.append({
                "role": "assistant",
                "content": assistant_content,
            })

            # Execute tool calls
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
            return "\n".join(text_blocks)


def _build_assistant_content(provider: str, response: object) -> list:
    """Build the assistant content list from the response."""
    if provider == "anthropic":
        return list(response.content)

    # OpenAI-compatible: construct tool call blocks
    content = []
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
        "--verbose", "-v",
        action="store_true",
        help="Show tool calls and stop reasons.",
    )
    return parser.parse_args(argv)


def _repl(provider: str, model: str, verbose: bool) -> None:
    """Run the interactive REPL."""
    print("obs-agent — Multi-Cloud Cost Triage Agent")
    print(f"Provider: {provider}  |  Model: {model}  |  Type 'quit' to exit.\n")
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
        answer = run(question, provider=provider, model=model, verbose=verbose)
        print(f"\nAgent: {answer}\n")


def main() -> None:
    args = _parse_args()
    model = args.model or PROVIDER_DEFAULT_MODELS.get(args.provider, DEFAULT_MODEL)

    if args.question:
        answer = run(
            args.question,
            provider=args.provider,
            model=model,
            verbose=args.verbose,
        )
        print(answer)
    else:
        _repl(provider=args.provider, model=model, verbose=args.verbose)


if __name__ == "__main__":
    main()
