"""Consumer contract tests with a stub HTTP boundary; no RAG server or LLM needed."""

import copy
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest

import documentation

EVIDENCE = {
    "chunks": [
        {
            "id": "reference/tagging-standard.md#1",
            "text": "Every resource requires an environment tag.",
            "source": "reference/tagging-standard.md",
            "file": "tagging-standard.md",
            "section": "Required tags",
            "distance": 0.2,
        }
    ],
    "corpus": {
        "status": "available",
        "corpus_sha256": "a" * 64,
        "ingested_at": "2026-10-31T12:00:00Z",
        "embedding_model": "all-MiniLM-L6-v2",
    },
}


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setenv("OBS_AGENT_RAG_URL", "http://rag.test:8001/")
    stub = Mock(side_effect=lambda *_args, **_kwargs: io.BytesIO(json.dumps(EVIDENCE).encode()))
    monkeypatch.setattr(documentation, "urlopen", stub)
    return stub


def test_consumer_posts_only_explicit_query_and_preserves_evidence(service, monkeypatch):
    monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", "/private/saved-reviews")
    result = json.loads(documentation.search_observability_docs("  Required tags?  ", n_results=3))
    assert result["chunks"][0] == {
        key: value for key, value in EVIDENCE["chunks"][0].items() if key != "file"
    }
    assert result["corpus"] == EVIDENCE["corpus"]
    request = service.call_args.args[0]
    assert request.full_url == "http://rag.test:8001/retrieve"
    assert request.get_method() == "POST"
    assert json.loads(request.data) == {"question": "Required tags?", "n_results": 3}
    assert request.get_header("Content-type") == "application/json"
    assert service.call_args.kwargs == {"timeout": 30}


def test_disabled_client_never_calls_network(service, monkeypatch):
    monkeypatch.delenv("OBS_AGENT_RAG_URL")
    assert "disabled" in documentation.search_observability_docs("Required tags?")
    service.assert_not_called()


@pytest.mark.parametrize(
    "question,count",
    [
        ("", 6),
        (" \n ", 6),
        ("x" * 4001, 6),
        (None, 6),
        ("q", 0),
        ("q", 21),
        ("q", True),
        ("q", "3"),
    ],
)
def test_invalid_tool_arguments_never_call_network(service, question, count):
    assert documentation.search_observability_docs(question, count).startswith("Error:")
    service.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "file:///private/data",
        "http://",
        "http://user:secret@rag.test",
        "http://rag.test?secret=value",
        "http://rag.test#fragment",
        "http://rag.test:bad",
    ],
)
def test_invalid_configuration_is_sanitized(service, monkeypatch, url):
    monkeypatch.setenv("OBS_AGENT_RAG_URL", url)
    result = documentation.search_observability_docs("Required tags?")
    assert result.startswith("Error:")
    assert "secret" not in result
    service.assert_not_called()


@pytest.mark.parametrize("corpus", [EVIDENCE["corpus"], {"status": "unavailable"}])
def test_no_matches_is_distinct_from_service_failure(service, corpus):
    service.side_effect = lambda *_a, **_k: io.BytesIO(
        json.dumps(
            {
                "chunks": [],
                "corpus": corpus,
            }
        ).encode()
    )
    assert json.loads(documentation.search_observability_docs("Unknown policy?")) == {
        "chunks": [],
        "corpus": corpus,
    }


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("private-host"),
        URLError("private-host"),
        HTTPError("http://private-host", 503, "private details", {}, None),
    ],
)
def test_transport_errors_are_safe_tool_results(service, error):
    service.side_effect = error
    result = documentation.search_observability_docs("Required tags?")
    assert result.startswith("Error:")
    assert "no policy evidence available" in result
    assert "private" not in result


@pytest.mark.parametrize(
    "content",
    [
        b"not JSON",
        b"\xff",
        b"[]",
        b'{"chunks": []}',
        b'{"chunks": [{}], "corpus": {"status": "unavailable"}}',
        b'{"chunks": [], "corpus": {"status": "available"}}',
        b"x" * 1_048_577,
    ],
)
def test_malformed_or_oversized_evidence_is_not_sent_to_model(service, content):
    service.side_effect = lambda *_a, **_k: io.BytesIO(content)
    assert documentation.search_observability_docs("Required tags?").startswith("Error:")


def test_requests_do_not_reuse_previous_evidence_or_forward_private_metadata(service):
    first = copy.deepcopy(EVIDENCE)
    first["private_inventory"] = "secret-review-a"
    first["chunks"][0]["reviewer_note"] = "secret-note"
    second = {"chunks": [], "corpus": {"status": "unavailable"}}
    service.side_effect = [
        io.BytesIO(json.dumps(first).encode()),
        io.BytesIO(json.dumps(second).encode()),
    ]
    assert "secret" not in documentation.search_observability_docs("Policy A?")
    assert json.loads(documentation.search_observability_docs("Policy B?")) == second
    assert [json.loads(call.args[0].data)["question"] for call in service.call_args_list] == [
        "Policy A?",
        "Policy B?",
    ]


def test_agent_dispatches_real_client_and_preserves_source_for_final_answer(service, monkeypatch):
    import agent
    from tools import TOOL_DEFINITIONS, TOOL_DISPATCH

    assert "search_observability_docs" in {item["name"] for item in TOOL_DEFINITIONS}
    assert "search_observability_docs" in TOOL_DISPATCH
    citation = "(source: reference/tagging-standard.md, section 'Required tags')."
    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_docs",
                                type="function",
                                function=SimpleNamespace(
                                    name="search_observability_docs",
                                    arguments=json.dumps({"question": "Required tags?"}),
                                ),
                            )
                        ],
                    ),
                )
            ],
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="An environment tag is required " + citation,
                        tool_calls=None,
                    ),
                )
            ]
        ),
    ]
    seen = []

    def create(_backend, messages, **_kwargs):
        seen.append(copy.deepcopy(messages))
        return responses.pop(0)

    monkeypatch.setattr(agent.Provider, "create", create)
    monkeypatch.setattr(agent.observability, "setup", lambda: None)
    answer, history = agent.run("What tags are required?", provider="openai", model="test-model")
    tool_result = next(item["content"] for item in seen[1] if item["role"] == "tool")
    assert json.loads(tool_result)["chunks"][0]["source"] == EVIDENCE["chunks"][0]["source"]
    assert json.loads(tool_result)["corpus"] == EVIDENCE["corpus"]
    assert citation in answer
    assert history[-1]["content"] == answer
    assert service.call_count == 1
