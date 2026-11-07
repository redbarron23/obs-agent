"""Compare caller tracing under bounded load using a local OTLP sink and fictional retrieval.

Run as a module from the checkout, or via the existing Compose smoke image with --live-rag.
Each scenario runs in a fresh process so SDK configuration and shutdown are independently bounded.
"""

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path

import policy_docs
import policy_tracing

REQUESTS = 256
CONCURRENCY = 4
WARMUP = 8
QUEUE = 64
BATCH = 16
HTTP_TIMEOUT = 0.5
QUESTION = "Which tags are required on Azure and GCP resources?"


def comparable(result):
    """Freshness check time can advance without changing the underlying retrieval evidence."""
    return {
        **result,
        "freshness": {
            k: v for k, v in (result.get("freshness") or {}).items() if k != "checked_at"
        },
    }


def scenario(mode, live_rag=False):
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    accepted = set()
    batches = []
    lock = threading.Lock()
    slow = threading.Event()
    exporting = threading.Event()

    class Collector(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            request = ExportTraceServiceRequest.FromString(
                self.rfile.read(int(self.headers["Content-Length"]))
            )
            spans = [
                s for r in request.resource_spans for scope in r.scope_spans for s in scope.spans
            ]
            assert self.path == "/v1/traces"
            with lock:
                batches.append(len(spans))
            if slow.is_set():
                exporting.set()
                time.sleep(1)  # Deliberately longer than the HTTP export deadline.
                return
            with lock:
                accepted.update(s.span_id.hex() for s in spans)
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    os.environ.update(
        {
            "OTEL_TRACES_EXPORTER": "none" if mode == "disabled" else "otlp",
            "OTEL_SDK_DISABLED": "false",
            "OTEL_TRACES_SAMPLER": "parentbased_always_on",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": f"http://127.0.0.1:{server.server_port}/v1/traces",
            "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf",
            "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT": str(HTTP_TIMEOUT),
            "OTEL_BSP_MAX_QUEUE_SIZE": str(QUEUE),
            "OTEL_BSP_MAX_EXPORT_BATCH_SIZE": str(BATCH),
            "OTEL_BSP_SCHEDULE_DELAY": "20",
        }
    )
    if not live_rag:
        payload = json.loads(
            (
                Path(__file__).resolve().parents[1] / "fixtures/rag_retrieve_tagging_standard.json"
            ).read_text()
        )
        send = policy_docs.stub_transport(200, payload)

        def transport(*args):
            time.sleep(0.04)  # Fixed retrieval-stub latency, identical in all scenarios.
            return send(*args)

        policy_docs.load_config = lambda: policy_docs.RagConfig("http://127.0.0.1:8765")
        policy_docs._transport = lambda: transport

    policy_tracing.provider.cache_clear()
    provider = policy_tracing.provider()
    queue = None
    if provider:
        # shortcut: inspect the locked SDK's queue, update this probe when SDK internals change.
        (processor,) = provider._active_span_processor._span_processors
        queue = processor._batch_processor._queue
        assert queue.maxlen == QUEUE

    def retrieve(_=None):
        started = time.perf_counter()
        result = policy_docs.retrieve(QUESTION, k=4)
        elapsed = time.perf_counter() - started
        assert comparable(result) == comparable(expected), (
            "retrieval evidence changed under tracing load"
        )
        return elapsed, len(queue) if queue is not None else 0

    try:
        expected = policy_docs.retrieve(QUESTION, k=4)
        assert expected["status"] == "ok" and expected["hits"], (
            "fictional retrieval did not succeed"
        )
        for _ in range(WARMUP):
            retrieve()
        if provider:
            provider.force_flush()
        with lock:
            accepted.clear()
            batches.clear()
        if mode == "slow":
            slow.set()
            for _ in range(BATCH):
                retrieve()
            assert exporting.wait(2), "slow export did not begin"

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            results = list(pool.map(retrieve, range(REQUESTS)))
        elapsed = time.perf_counter() - started
        durations = sorted(duration for duration, _ in results)
        high_water = max(size for _, size in results)
        assert high_water <= QUEUE
        assert durations[-1] < 5 and elapsed < 60, "request availability budget exceeded"
        if mode == "slow" and not live_rag:
            assert high_water == QUEUE, "fixture workload did not saturate the queue"

        @policy_tracing.retrieval
        def application_failure():
            raise original

        original = RuntimeError("fictional application failure")
        try:
            application_failure()
        except RuntimeError as exc:
            assert exc is original, "telemetry replaced the application exception"
        else:
            raise AssertionError("application exception was swallowed")

        # Flush while still slow: finite queues and HTTP deadlines must let this finish.
        flush_started = time.perf_counter()
        if provider:
            provider.force_flush()
        flush_seconds = time.perf_counter() - flush_started
        assert flush_seconds < 5, "flush budget exceeded"
        with lock:
            delivered = len(accepted)
            max_batch = max(batches, default=0)
        assert max_batch <= BATCH
        generated = REQUESTS + 1 + (BATCH if mode == "slow" else 0)
        assert delivered <= generated
        if mode == "healthy":
            assert delivered == generated, "healthy collector lost spans"
        if mode == "disabled":
            assert not batches and delivered == 0
        recovery = False
        shutdown_queued = 0
        if mode == "slow":
            assert delivered == 0, "slow sink unexpectedly accepted a batch"
            slow.clear()
            retrieve()
            provider.force_flush()
            with lock:
                assert len(accepted) == 1, "recovery did not deliver exactly one fresh span"
            recovery = True

            @policy_tracing.retrieval
            def shutdown_probe():
                return {"status": "ok"}

            slow.set()
            exporting.clear()
            for _ in range(BATCH):
                shutdown_probe()
            assert exporting.wait(2), "shutdown probe did not begin exporting"
            for _ in range(QUEUE):
                shutdown_probe()
            shutdown_queued = len(queue)
            assert 0 < shutdown_queued <= QUEUE

        shutdown_started = time.perf_counter()
        if provider:
            provider.shutdown()
        shutdown_seconds = time.perf_counter() - shutdown_started
        assert shutdown_seconds < 5, "shutdown budget exceeded"
        provider = None
        return {
            "scenario": mode,
            "requests": REQUESTS,
            "concurrency": CONCURRENCY,
            "warmup": WARMUP + 1,
            "successful_results": REQUESTS,
            "original_exception_preserved": True,
            "elapsed_seconds": elapsed,
            "latency_ms": {
                "p50": durations[len(durations) // 2] * 1000,
                "p95": durations[math.ceil(len(durations) * 0.95) - 1] * 1000,
                "max": durations[-1] * 1000,
            },
            "queue_capacity": QUEUE,
            "queue_high_water_observed": high_water,
            "max_received_batch": max_batch,
            "generated_spans": generated if mode != "disabled" else 0,
            "accepted_spans": delivered,
            "undelivered_spans": generated - delivered if mode != "disabled" else 0,
            "flush_seconds": flush_seconds,
            "shutdown_seconds": shutdown_seconds,
            "queued_before_shutdown": shutdown_queued,
            "shutdown_probe_spans": BATCH + QUEUE if mode == "slow" else 0,
            "fresh_span_after_recovery": recovery,
        }
    finally:
        if provider:
            provider.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def check(live_rag=False):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("OTEL_", "MLFLOW_")) and k != "OBS_AGENT_MLFLOW"
    }
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    env["NO_PROXY"] = "127.0.0.1,localhost"
    results = []
    for mode in ("disabled", "healthy", "slow"):
        command = [sys.executable, str(Path(__file__).resolve()), "--scenario", mode]
        if live_rag:
            command.append("--live-rag")
        child = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
        assert child.returncode == 0, f"{mode} process failed:\n{child.stderr[-4000:]}"
        results.append(json.loads(child.stdout))
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: version(name)
            for name in ("opentelemetry-sdk", "opentelemetry-exporter-otlp-proto-http")
        },
        "retrieval": "configured real RAG service"
        if live_rag
        else "fictional fixture; injected 40ms delay",
        "scope": "caller SDK queue/export only; provider tracing unchanged",
        "collector": "local OTLP HTTP/protobuf sink; slow mode injects 1s per batch",
        "limits": {
            "request_max_seconds": 5,
            "workload_seconds": 60,
            "flush_seconds": 5,
            "shutdown_seconds": 5,
            "process_seconds": 90,
            "http_export_timeout_seconds": HTTP_TIMEOUT,
        },
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-rag", action="store_true")
    parser.add_argument(
        "--scenario", choices=("disabled", "healthy", "slow"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    print(
        json.dumps(
            scenario(args.scenario, args.live_rag) if args.scenario else check(args.live_rag),
            indent=2,
        )
    )
