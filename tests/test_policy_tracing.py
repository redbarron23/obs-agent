"""Correlation changes headers and metadata, never retrieval behavior or content."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

import policy_docs
import policy_tracing

FIXTURE = json.loads(
    (Path(__file__).parents[1] / "fixtures/rag_retrieve_tagging_standard.json").read_text()
)


@pytest.fixture
def recording(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(policy_tracing, "provider", lambda: provider)
    yield provider, exporter.get_finished_spans
    provider.shutdown()


@pytest.mark.parametrize(
    "outcome", ["ok", "no_match", "unauthorized", "service_error", "timeout", "unreachable"]
)
def test_outcomes_and_headers_are_safe(recording, monkeypatch, outcome):
    provider, spans = recording
    payload = json.loads(json.dumps(FIXTURE))
    if outcome == "no_match":
        payload["hits"] = []
    captured = []

    def transport(url, body, headers, timeout):
        captured.append(headers)
        assert json.loads(body)["question"] == "PRIVATE QUESTION"
        assert timeout == 2
        if outcome == "timeout":
            raise TimeoutError("PRIVATE EXCEPTION")
        if outcome == "unreachable":
            raise OSError("PRIVATE EXCEPTION")
        code = {"unauthorized": 401, "service_error": 500}.get(outcome, 200)
        return code, json.dumps(payload).encode()

    monkeypatch.setattr(
        policy_docs,
        "load_config",
        lambda: policy_docs.RagConfig(
            "http://127.0.0.1:8765", api_key="PRIVATE CREDENTIAL", timeout=2
        ),
    )
    monkeypatch.setattr(policy_docs, "_transport", lambda: transport)
    result = policy_docs.retrieve("PRIVATE QUESTION")
    assert result["status"] == outcome
    (span,) = spans()
    assert span.attributes["rag.client.outcome"] == outcome
    assert span.status.status_code == (
        StatusCode.UNSET if outcome in ("ok", "no_match") else StatusCode.ERROR
    )
    context = TraceContextTextMapPropagator().extract(captured[0])
    from opentelemetry.trace import get_current_span

    parent = get_current_span(context).get_span_context()
    assert parent.span_id == span.context.span_id and parent.trace_id == span.context.trace_id
    assert captured[0]["Authorization"] == "Bearer PRIVATE CREDENTIAL"
    assert "baggage" not in captured[0]
    assert "PRIVATE" not in span.to_json()
    assert not span.events


def test_concurrent_calls_have_distinct_context(recording, monkeypatch):
    _, spans = recording
    monkeypatch.setattr(
        policy_docs, "load_config", lambda: policy_docs.RagConfig("http://127.0.0.1:8765")
    )
    headers = []

    def transport(url, body, carrier, timeout):
        headers.append(carrier["traceparent"])
        return 200, json.dumps(FIXTURE).encode()

    monkeypatch.setattr(policy_docs, "_transport", lambda: transport)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert all(r["status"] == "ok" for r in pool.map(policy_docs.retrieve, ["one", "two"]))
    assert len(set(headers)) == 2
    assert len({s.context.trace_id for s in spans()}) == 2


def test_trace_failures_leave_result_unchanged(monkeypatch):
    config = policy_docs.RagConfig("http://127.0.0.1:8765")
    monkeypatch.setattr(policy_docs, "load_config", lambda: config)
    monkeypatch.setattr(policy_docs, "_transport", lambda: policy_docs.stub_transport(200, FIXTURE))
    provider = Mock()
    provider.get_tracer.side_effect = RuntimeError("PRIVATE SDK FAILURE")
    monkeypatch.setattr(policy_tracing, "provider", lambda: provider)
    assert policy_docs.retrieve("question")["status"] == "ok"


@pytest.mark.parametrize("exporter,disabled", [(None, False), ("none", False), ("otlp", True)])
def test_disabled_tracing_sends_no_context(monkeypatch, exporter, disabled):
    policy_tracing.provider.cache_clear()
    if exporter is None:
        monkeypatch.delenv("OTEL_TRACES_EXPORTER", raising=False)
    else:
        monkeypatch.setenv("OTEL_TRACES_EXPORTER", exporter)
    monkeypatch.setenv("OTEL_SDK_DISABLED", str(disabled).lower())
    monkeypatch.setattr(
        policy_docs, "load_config", lambda: policy_docs.RagConfig("http://127.0.0.1:8765")
    )

    def transport(url, body, headers, timeout):
        assert "traceparent" not in headers and "tracestate" not in headers
        return 200, json.dumps(FIXTURE).encode()

    monkeypatch.setattr(policy_docs, "_transport", lambda: transport)
    try:
        assert policy_docs.retrieve("question")["status"] == "ok"
    finally:
        policy_tracing.provider.cache_clear()


def test_only_w3c_context_propagates_from_parent(recording, monkeypatch):
    from opentelemetry import baggage, context, trace

    _, spans = recording
    config = policy_docs.RagConfig("http://127.0.0.1:8765", api_key="PRIVATE KEY")
    monkeypatch.setattr(policy_docs, "load_config", lambda: config)
    parent = trace.NonRecordingSpan(
        trace.SpanContext(
            trace_id=123,
            span_id=456,
            is_remote=True,
            trace_flags=trace.TraceFlags(1),
            trace_state=trace.TraceState([("vendor", "opaque")]),
        )
    )
    ctx = baggage.set_baggage("reviewer", "PRIVATE REVIEWER", trace.set_span_in_context(parent))
    token = context.attach(ctx)

    def transport(url, body, headers, timeout):
        assert headers["tracestate"] == "vendor=opaque"
        assert "baggage" not in headers and "PRIVATE REVIEWER" not in str(headers)
        return 200, json.dumps(FIXTURE).encode()

    monkeypatch.setattr(policy_docs, "_transport", lambda: transport)
    try:
        assert policy_docs.retrieve("question")["status"] == "ok"
    finally:
        context.detach(token)
    assert spans()[0].parent.span_id == 456


def test_missing_optional_sdk_is_safe(monkeypatch, caplog):
    import builtins

    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name.startswith("opentelemetry.sdk"):
            raise ImportError("PRIVATE IMPORT ERROR")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "console")
    policy_tracing.provider.cache_clear()
    try:
        assert policy_tracing.provider() is None
        assert "PRIVATE" not in caplog.text
    finally:
        policy_tracing.provider.cache_clear()


def test_span_finish_failure_does_not_replace_answer(monkeypatch, caplog):
    fake = Mock()
    manager = fake.get_tracer.return_value.start_as_current_span.return_value
    manager.__enter__ = Mock(return_value=Mock())
    manager.__exit__ = Mock(side_effect=RuntimeError("PRIVATE FINISH FAILURE"))
    monkeypatch.setattr(policy_tracing, "provider", lambda: fake)
    monkeypatch.setattr(
        policy_docs, "load_config", lambda: policy_docs.RagConfig("http://127.0.0.1:8765")
    )
    monkeypatch.setattr(policy_docs, "_transport", lambda: policy_docs.stub_transport(200, FIXTURE))
    assert policy_docs.retrieve("question")["status"] == "ok"
    assert "PRIVATE" not in caplog.text
