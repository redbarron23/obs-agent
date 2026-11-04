"""Verify the real caller/provider trace graph in the opt-in local Jaeger collector."""

import json
import os
import time
from urllib.request import urlopen

import policy_docs
import policy_tracing


def attributes(span):
    return {a["key"]: next(iter(a["value"].values())) for a in span.get("attributes", [])}


def traces(payload):
    result = {}
    for resource in payload.get("result", {}).get("resourceSpans", []):
        service = attributes(resource.get("resource", {})).get("service.name")
        for scope in resource.get("scopeSpans", []):
            for span in scope.get("spans", []):
                result.setdefault(span["traceId"], []).append({**span, "service.name": service})
    return result


def verify(spans):
    assert len(spans) == 3, "expected exactly caller/question/retrieve spans"
    named = {s["name"]: s for s in spans}
    assert set(named) == {"obs.policy.retrieve", "rag.question", "rag.retrieve"}
    caller, root, retrieval = (
        named[n] for n in ("obs.policy.retrieve", "rag.question", "rag.retrieve")
    )
    assert len({s["traceId"] for s in spans}) == 1
    assert not caller.get("parentSpanId")
    assert root.get("parentSpanId") == caller["spanId"], "question has no caller parent"
    assert retrieval.get("parentSpanId") == root["spanId"], "retrieval has no question parent"
    assert caller["service.name"] == os.environ.get(
        "JAEGER_CALLER_SERVICE_NAME", "obs-agent-policy"
    )
    rag_service = os.environ.get("JAEGER_SERVICE_NAME", "obs-rag-retrieval")
    assert root["service.name"] == retrieval["service.name"] == rag_service
    assert caller["service.name"] != rag_service
    assert attributes(caller)["rag.client.outcome"] == "ok"
    assert attributes(root)["rag.generation.called"] is False
    values = attributes(retrieval)
    assert int(values["rag.retrieval.hit_count"]) > 0
    assert values["rag.retrieval.mode"] == "hybrid"
    assert int(values["rag.retrieval.k"]) == 4
    assert "reference/tagging-standard.md" in [
        v for k, v in values.items() if k.endswith(".source")
    ]
    assert any(k.endswith(".distance") for k in values)


def check():
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import format_trace_id

    jaeger = os.environ["JAEGER_URL"].rstrip("/")
    deadline = time.monotonic() + 30
    while True:
        try:
            with urlopen(jaeger + "/api/v3/services", timeout=2):
                break
        except OSError:
            assert time.monotonic() < deadline, "Jaeger query API did not become ready"
            time.sleep(0.2)
    provider = policy_tracing.provider()
    assert provider is not None, "caller tracing is disabled"
    recorded = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(recorded))
    question = "Which tags are required on Azure and GCP resources?"
    result = policy_docs.retrieve(question, k=4)
    assert result["status"] == "ok", result["status"]
    (caller,) = recorded.get_finished_spans()
    trace_id = format_trace_id(caller.context.trace_id)
    assert provider.force_flush(timeout_millis=5000), "caller export did not flush"
    deadline = time.monotonic() + 30
    while True:
        try:
            with urlopen(jaeger + "/api/v3/traces/" + trace_id, timeout=2) as response:
                found = traces(json.load(response)).get(trace_id, [])
        except OSError:
            found = []
        if len(found) == 3:
            verify(found)
            serialized = json.dumps(found)
            for text in (question, os.environ["RAG_API_KEY"], *[h["text"] for h in result["hits"]]):
                assert text not in serialized, "trace contains content or credentials"
            print(
                f"PASS: distributed obs-agent/RAG trace {trace_id}; "
                "three linked spans, two services"
            )
            return
        assert time.monotonic() < deadline, "complete distributed trace not found in Jaeger"
        time.sleep(0.2)


if __name__ == "__main__":
    check()
