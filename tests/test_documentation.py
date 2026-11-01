"""Consumer contract tests with a stub HTTP boundary; no RAG server or LLM needed."""

import copy
import io
import json
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError, URLError
from urllib.request import BaseHandler
from urllib.response import addinfourl

import pytest

import documentation

# Captured from the merged RAG API; see docs/testing.md for its pinned revision.
EVIDENCE = json.loads((Path(__file__).parent / "fixtures/rag_retrieve_response.json").read_text())


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setenv("OBS_AGENT_RAG_URL", "http://rag.test:8765/")
    monkeypatch.delenv("OBS_AGENT_RAG_API_KEY", raising=False)
    stub = Mock(side_effect=lambda *_args, **_kwargs: io.BytesIO(json.dumps(EVIDENCE).encode()))
    monkeypatch.setattr(documentation._opener, "open", stub)
    return stub


def test_consumer_posts_only_explicit_query_and_preserves_evidence(service, monkeypatch):
    monkeypatch.setenv("OBS_AGENT_REVIEW_DIR", "/private/saved-reviews")
    result = json.loads(documentation.search_observability_docs("  Required tags?  ", k=3))
    assert result == {key: value for key, value in EVIDENCE.items() if key != "query"}
    assert result["corpus"] == EVIDENCE["corpus"]
    request = service.call_args.args[0]
    assert request.full_url == "http://rag.test:8765/retrieve"
    assert request.get_method() == "POST"
    assert json.loads(request.data) == {"question": "Required tags?", "k": 3}
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("Authorization") is None
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
        ("x" * 2001, 6),
        (None, 6),
        ("q\x00", 6),
        ("q", 0),
        ("q", 21),
        ("q", True),
        ("q", "3"),
    ],
)
def test_invalid_tool_arguments_never_call_network(service, question, count):
    assert documentation.search_observability_docs(question, count).startswith("Error:")
    service.assert_not_called()


def test_optional_bearer_key_is_only_sent_in_header(service, monkeypatch):
    monkeypatch.setenv("OBS_AGENT_RAG_API_KEY", "test-secret")
    result = documentation.search_observability_docs("Required tags?")
    request = service.call_args.args[0]
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert "test-secret" not in request.data.decode()
    assert "test-secret" not in result


def test_redirect_does_not_forward_question_or_bearer_credentials(monkeypatch):
    calls = []

    class RedirectServer(BaseHandler):
        handler_order = 100

        def http_open(self, request):
            calls.append(request.full_url)
            headers = Message()
            headers["Location"] = "http://other.test/retrieve"
            response = addinfourl(io.BytesIO(b""), headers, request.full_url, 302)
            response.msg = "Found"
            return response

    monkeypatch.setattr(
        documentation,
        "_opener",
        documentation.build_opener(
            documentation._NoRedirect(),
            RedirectServer(),
        ),
    )
    monkeypatch.setenv("OBS_AGENT_RAG_URL", "http://rag.test:8765")
    monkeypatch.setenv("OBS_AGENT_RAG_API_KEY", "test-secret")
    assert "HTTP 302" in documentation.search_observability_docs("Private policy question?")
    assert calls == ["http://rag.test:8765/retrieve"]


def test_unicode_question_at_server_limit_fits_request_body(service):
    question = "\U0001f30d" * 2000
    assert not documentation.search_observability_docs(question, k=20).startswith("Error:")
    assert json.loads(service.call_args.args[0].data) == {"question": question, "k": 20}
    assert len(service.call_args.args[0].data) <= 16 * 1024


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


@pytest.mark.parametrize("provenance", ["available", "unavailable"])
def test_no_matches_is_distinct_from_service_failure(service, provenance):
    evidence = copy.deepcopy(EVIDENCE)
    evidence["hits"] = []
    if provenance == "unavailable":
        evidence["corpus"] = {
            "provenance": "unavailable",
            "embedding_model": "all-MiniLM-L6-v2",
            "collection": "legacy",
            "chunk_count": 1,
        }
    service.side_effect = lambda *_a, **_k: io.BytesIO(json.dumps(evidence).encode())
    assert json.loads(documentation.search_observability_docs("Unknown policy?")) == {
        key: value for key, value in evidence.items() if key != "query"
    }


@pytest.mark.parametrize("status", ["current", "stale", "unavailable"])
def test_source_freshness_survives_for_the_agent(service, status):
    evidence = copy.deepcopy(EVIDENCE)
    evidence["freshness"]["status"] = status
    if status == "stale":
        evidence["freshness"]["modified"] = ["reference/tagging-standard.md"]
    if status == "unavailable":
        evidence["freshness"] = {
            "status": status,
            "checked_at": EVIDENCE["freshness"]["checked_at"],
            "reason": "Source documents could not be compared with the index.",
        }
    service.side_effect = lambda *_a, **_k: io.BytesIO(json.dumps(evidence).encode())
    assert (
        json.loads(documentation.search_observability_docs("Required tags?"))["freshness"]
        == (evidence["freshness"])
    )


@pytest.mark.parametrize("field,value", [("content_trust", "trusted"), ("generated", True)])
def test_unexpected_trust_or_generation_claim_is_rejected(service, field, value):
    evidence = copy.deepcopy(EVIDENCE)
    evidence[field] = value
    service.side_effect = lambda *_a, **_k: io.BytesIO(json.dumps(evidence).encode())
    assert "invalid documentation response" in documentation.search_observability_docs("Policy?")


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("private-host"),
        URLError("private-host"),
        HTTPError("http://private-host", 503, "private details", {}, None),
        HTTPError("http://private-host", 401, "private details", {}, None),
        HTTPError("http://private-host", 302, "private details", {}, None),
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
        b'{"hits": []}',
        b'{"hits": [{}], "corpus": {"provenance": "unavailable"}}',
        b'{"hits": [], "corpus": {"provenance": "available"}}',
        b'{"chunks": [], "corpus": {"status": "unavailable"}}',
        b"x" * 1_048_577,
    ],
)
def test_malformed_or_oversized_evidence_is_not_sent_to_model(service, content):
    service.side_effect = lambda *_a, **_k: io.BytesIO(content)
    assert documentation.search_observability_docs("Required tags?").startswith("Error:")


def test_requests_do_not_reuse_previous_evidence_or_forward_private_metadata(service):
    first = copy.deepcopy(EVIDENCE)
    first["private_inventory"] = "secret-review-a"
    first["hits"][0]["reviewer_note"] = "secret-note"
    second = copy.deepcopy(EVIDENCE)
    second["hits"] = []
    second.pop("query")
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
    hit = EVIDENCE["hits"][0]
    citation = f"(source: {hit['source']}, section '{hit['section']}')."
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
    assert json.loads(tool_result)["hits"][0]["source"] == EVIDENCE["hits"][0]["source"]
    assert json.loads(tool_result)["corpus"] == EVIDENCE["corpus"]
    assert citation in answer
    assert history[-1]["content"] == answer
    assert service.call_count == 1
