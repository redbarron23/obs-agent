"""Optional content-free OpenTelemetry spans for the documentation HTTP caller."""

import logging
import os
import sys
from contextvars import ContextVar
from functools import lru_cache, wraps
from time import perf_counter

_log = logging.getLogger(__name__)
_current = ContextVar("obs_policy_span", default=None)


@lru_cache(maxsize=1)
def provider():
    exporter = os.environ.get("OTEL_TRACES_EXPORTER", "").strip().lower()
    if exporter in ("", "none") or os.environ.get("OTEL_SDK_DISABLED", "").lower() == "true":
        return None
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        if exporter == "console":
            target = ConsoleSpanExporter(out=sys.stderr)
        elif exporter == "otlp":
            protocol = os.environ.get(
                "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
                os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"),
            )
            if protocol != "http/protobuf":
                raise ValueError("Only HTTP/protobuf is supported")
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            target = OTLPSpanExporter()
        else:
            raise ValueError("Unsupported exporter")
        result = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": os.environ.get("OTEL_SERVICE_NAME", "obs-agent-policy"),
                }
            )
        )
        result.add_span_processor(BatchSpanProcessor(target))
        return result
    except Exception as exc:
        _log.warning("Policy tracing disabled (%s).", type(exc).__name__)
        return None


def attributes(values):
    span = _current.get()
    if span is not None:
        try:
            span.set_attributes(values)
        except Exception:
            _log.warning("Could not record policy trace attributes.")


def inject(headers):
    span = _current.get()
    if span is not None:
        try:
            from opentelemetry import trace
            from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

            TraceContextTextMapPropagator().inject(headers, context=trace.set_span_in_context(span))
        except Exception:
            _log.warning("Could not inject policy trace context.")


def retrieval(function):
    @wraps(function)
    def call(*args, **kwargs):
        manager = span = None
        try:
            configured = provider()
            if configured is not None:
                from opentelemetry.trace import SpanKind

                manager = configured.get_tracer("obs-agent-policy").start_as_current_span(
                    "obs.policy.retrieve",
                    kind=SpanKind.CLIENT,
                    record_exception=False,
                    set_status_on_exception=False,
                )
                span = manager.__enter__()
        except Exception:
            manager = span = None
            _log.warning("Could not start policy trace.")
        token = _current.set(span)
        started = perf_counter()
        try:
            result = function(*args, **kwargs)
            attributes({"rag.client.outcome": result["status"]})
            if result["status"] not in ("ok", "no_match", "disabled") and span is not None:
                try:
                    from opentelemetry.trace import StatusCode

                    span.set_status(StatusCode.ERROR)
                except Exception:
                    _log.warning("Could not record policy trace status.")
            return result
        finally:
            attributes({"rag.duration_ms": (perf_counter() - started) * 1000})
            _current.reset(token)
            if manager is not None:
                try:
                    manager.__exit__(None, None, None)
                except Exception:
                    _log.warning("Could not finish policy trace.")

    return call
