"""Fail closed when Jaeger returns the wrong graph or retrieval evidence."""

from copy import deepcopy

import pytest

from scripts.telemetry_stack_smoke import traces, verify


@pytest.fixture
def spans():
    def attrs(**values):
        return [{"key": key, "value": {"stringValue": value}} for key, value in values.items()]

    root = {
        "name": "rag.question",
        "spanId": "root",
        "traceId": "trace",
        "attributes": [
            {"key": "rag.generation.called", "value": {"boolValue": False}},
        ],
    }
    child = {
        "name": "rag.retrieve",
        "spanId": "child",
        "parentSpanId": "root",
        "traceId": "trace",
        "attributes": attrs(
            **{
                "rag.retrieval.mode": "hybrid",
                "rag.retrieval.k": "4",
                "rag.retrieval.hit_count": "1",
                "rag.retrieval.hit.1.source": "reference/tagging-standard.md",
                "rag.retrieval.hit.1.distance": "0.2",
            }
        ),
    }
    return [root, child]


def test_complete_retrieval_trace(spans):
    verify(spans)
    payload = {"result": {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}}
    assert traces(payload) == {"trace": spans}


@pytest.mark.parametrize("damage", ["parent", "source", "generation", "missing"])
def test_wrong_trace_is_rejected(spans, damage):
    broken = deepcopy(spans)
    if damage == "parent":
        broken[1]["parentSpanId"] = "unrelated"
    elif damage == "source":
        broken[1]["attributes"][-2]["value"] = {"stringValue": "unrelated.md"}
    elif damage == "generation":
        broken[0]["attributes"][0]["value"] = {"boolValue": True}
    else:
        broken.pop()
    with pytest.raises(AssertionError):
        verify(broken)
