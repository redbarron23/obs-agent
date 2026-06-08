#!/usr/bin/env python3
"""Multi-Cloud Cost Triage Agent — Claude agent loop with tool-use.

Usage
-----
Interactive REPL:
    python agent.py

Single question (scriptable):
    python agent.py -q "Which Azure subscription has the highest overage?"

Use a different model:
    python agent.py -q "Show me the top 3 GCP projects" --model claude-sonnet-4-6

Run verbosely (see tool calls):
    python agent.py -q "Any cost spikes?" --verbose
"""

import argparse
import sys

import anthropic
from tools import TOOL_DEFINITIONS, TOOL_DISPATCH

DEFAULT_MODEL = "claude-sonnet-4-6"

SYSTEM = """You are a multi-cloud cost triage assistant.
You have access to cloud logging billing data for Azure and GCP.
When answering questions, always call the relevant tools to get actual numbers.
Be concise and highlight the most important findings."""


def run(
    question: str,
    *,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> str:
    """Run a single question through the agent loop and return the answer."""
    client = anthropic.Anthropic()
    messages: list[dict] = [{"role": "user", "content": question}]

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        if verbose:
            print(f"  [stop_reason: {response.stop_reason}]", file=sys.stderr)

        text_blocks = [
            b.text for b in response.content if hasattr(b, "text")
        ]

        if response.stop_reason == "end_turn":
            return "\n".join(text_blocks)

        if response.stop_reason == "tool_use":
            # Append assistant message with tool-use blocks
            messages.append({"role": "assistant", "content": response.content})

            # Execute tool calls and collect results
            tool_results = []
            for block in response.content:
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
            # Unexpected stop reason — return what we have
            return "\n".join(text_blocks)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Cloud Cost Triage Agent",
    )
    parser.add_argument(
        "-q", "--question",
        help="Single question to answer (omit for interactive REPL).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Claude model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show tool calls and stop reasons.",
    )
    return parser.parse_args(argv)


def _repl(model: str, verbose: bool) -> None:
    """Run the interactive REPL."""
    print("obs-agent — Multi-Cloud Cost Triage Agent")
    print(f"Model: {model}  |  Type 'quit' to exit.\n")
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
        answer = run(question, model=model, verbose=verbose)
        print(f"\nAgent: {answer}\n")


def main() -> None:
    args = _parse_args()

    if args.question:
        answer = run(
            args.question,
            model=args.model,
            verbose=args.verbose,
        )
        print(answer)
    else:
        _repl(model=args.model, verbose=args.verbose)


if __name__ == "__main__":
    main()
