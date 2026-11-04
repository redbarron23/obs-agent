"""Verify one production-client retrieval in the opt-in local Jaeger collector."""

import json
import os
import time
from datetime import UTC, datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import policy_docs


def attributes(span):
    return {a["key"]: next(iter(a["value"].values())) for a in span.get("attributes", [])}


def traces(payload):
    result = {}
    for resource in payload.get("result", {}).get("resourceSpans", []):
        for scope in resource.get("scopeSpans", []):
            for span in scope.get("spans", []):
                result.setdefault(span["traceId"], []).append(span)
    return result


def verify(spans):
    assert len(spans) == 2, "expected exactly question/retrieve spans"
    named = {s["name"]: s for s in spans}
    assert set(named) == {"rag.question", "rag.retrieve"}, "wrong span names"
    root, retrieval = named["rag.question"], named["rag.retrieve"]
    assert retrieval.get("parentSpanId") == root["spanId"], "retrieval parent is not the question"
    assert attributes(root)["rag.generation.called"] is False, "retrieval invoked generation"
    values = attributes(retrieval)
    assert int(values["rag.retrieval.hit_count"]) > 0, "trace has no retrieval hits"
    assert values["rag.retrieval.mode"] == "hybrid"
    assert int(values["rag.retrieval.k"]) == 4
    assert "reference/tagging-standard.md" in [
        value for key, value in values.items() if key.endswith(".source")
    ], "trace is missing the retrieved tagging-policy source"
    assert any(key.endswith(".distance") for key in values), "trace distances are missing"


def check():
    jaeger = os.environ["JAEGER_URL"].rstrip("/")
    service = os.environ["JAEGER_SERVICE_NAME"]
    start = datetime.now(UTC)
    params = urlencode(
        {
            "query.service_name": service,
            "query.start_time_min": start.isoformat(),
            "query.start_time_max": (start + timedelta(minutes=2)).isoformat(),
            "query.search_depth": "20",
        }
    )
    endpoint = jaeger + "/api/v3/traces?" + params

    def read():
        try:
            with urlopen(endpoint, timeout=2) as response:
                return traces(json.load(response))
        except HTTPError as exc:
            if exc.code == 404:
                return {}
            raise

    deadline = time.monotonic() + 30
    while True:
        try:
            baseline = set(read())
            break
        except URLError:
            assert time.monotonic() < deadline, "Jaeger query API did not become ready"
            time.sleep(0.2)
    question = "Which tags are required on Azure and GCP resources?"
    result = policy_docs.retrieve(question, k=4)
    assert result["status"] == "ok", result["status"]
    deadline = time.monotonic() + 30
    while True:
        found = read()
        new = set(found) - baseline
        assert len(new) <= 1, "concurrent queries found; rerun without other policy requests"
        if new:
            trace_id = new.pop()
            spans = found[trace_id]
            # A batch can be visible before the complete question trace is stored.
            if len(spans) == 2:
                verify(spans)
                serialized = json.dumps(spans)
                for text in (
                    question,
                    os.environ["RAG_API_KEY"],
                    *[h["text"] for h in result["hits"]],
                ):
                    assert text not in serialized, "trace contains request content or credentials"
                print(f"PASS: real obs-agent retrieval verified in Jaeger trace {trace_id}")
                return
        assert time.monotonic() < deadline, "new complete retrieval trace not found in Jaeger"
        time.sleep(0.2)


if __name__ == "__main__":
    check()
