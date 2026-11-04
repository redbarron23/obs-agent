"""Fail closed when Jaeger returns the wrong graph or retrieval evidence."""

from copy import deepcopy

import pytest

from scripts.telemetry_stack_smoke import traces, verify


@pytest.fixture
def spans():
    def attrs(**values):
        return [{"key": key, "value": {"stringValue": value}} for key, value in values.items()]

    caller = {
        "name": "obs.policy.retrieve",
        "spanId": "caller",
        "traceId": "trace",
        "service.name": "obs-agent-policy",
        "attributes": attrs(**{"rag.client.outcome": "ok"}),
    }
    root = {
        "name": "rag.question",
        "spanId": "root",
        "parentSpanId": "caller",
        "service.name": "obs-rag-retrieval",
        "traceId": "trace",
        "attributes": [
            {"key": "rag.generation.called", "value": {"boolValue": False}},
        ],
    }
    child = {
        "name": "rag.retrieve",
        "spanId": "child",
        "service.name": "obs-rag-retrieval",
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
    return [root, child, caller]


def test_complete_retrieval_trace(spans):
    verify(spans)
    payload = {
        "result": {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": s["service.name"]}}
                        ]
                    },
                    "scopeSpans": [{"spans": [s]}],
                }
                for s in spans
            ]
        }
    }
    assert traces(payload) == {"trace": spans}


@pytest.mark.parametrize(
    "damage", ["parent", "source", "generation", "missing", "caller_parent", "service", "trace_id"]
)
def test_wrong_trace_is_rejected(spans, damage):
    broken = deepcopy(spans)
    if damage == "parent":
        broken[1]["parentSpanId"] = "unrelated"
    elif damage == "source":
        broken[1]["attributes"][-2]["value"] = {"stringValue": "unrelated.md"}
    elif damage == "generation":
        broken[0]["attributes"][0]["value"] = {"boolValue": True}
    elif damage == "caller_parent":
        broken[0]["parentSpanId"] = "unrelated"
    elif damage == "service":
        broken[-1]["service.name"] = "obs-rag-retrieval"
    elif damage == "trace_id":
        broken[-1]["traceId"] = "unrelated"
    else:
        broken.pop()
    with pytest.raises(AssertionError):
        verify(broken)
