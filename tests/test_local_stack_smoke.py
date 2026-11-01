"""Check smoke-test failure detection without Docker, sockets or a provider."""

import io
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import policy_docs
from scripts import local_stack_smoke as smoke


@pytest.fixture
def stack(monkeypatch):
    monkeypatch.setenv("OBS_AGENT_API_KEY", "local-test-key")
    monkeypatch.setattr(
        smoke,
        "urlopen",
        Mock(
            side_effect=[
                nullcontext(SimpleNamespace(status=200)),
                io.BytesIO(b'{"total": 320}'),
            ]
        ),
    )
    result = {
        "status": "ok",
        "hits": [{"source": "reference/tagging-standard.md", "section": "Required tags"}],
        "corpus": {"provenance": "available", "corpus_sha256": "test-hash"},
        "freshness": {"status": "current"},
    }
    retrieve = Mock(side_effect=[result, {"status": "no_match"}, {"status": "unauthorized"}])
    monkeypatch.setattr(policy_docs, "retrieve", retrieve)
    monkeypatch.setattr(
        policy_docs,
        "load_config",
        lambda: policy_docs.RagConfig(url="http://127.0.0.1:8765", api_key="test-key"),
    )
    return retrieve, result


def test_smoke_checks_the_live_client_contract(stack):
    retrieve, _ = stack
    smoke.check()
    assert retrieve.call_count == 3


def test_smoke_rejects_wrong_source(stack):
    _, result = stack
    result["hits"] = []
    with pytest.raises(AssertionError, match="source/section missing"):
        smoke.check()


def test_outage_simulation_requires_an_actual_outage(stack):
    with pytest.raises(AssertionError, match="ok"):
        smoke.check(rag_down=True)


def test_outage_simulation_keeps_inventory_check(stack):
    retrieve, _ = stack
    retrieve.side_effect = [{"status": "unreachable"}]
    smoke.check(rag_down=True)
