"""Check the running local Compose stack without an LLM or cloud credentials."""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

import policy_docs


def check(rag_down: bool = False, stale: bool = False) -> None:
    with urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
        assert response.status == 200, "obs-agent API is not healthy"
    request = Request(
        "http://127.0.0.1:8000/labels/coverage",
        headers={"X-API-Key": os.environ["OBS_AGENT_API_KEY"]},
    )
    with urlopen(request, timeout=30) as response:
        assert json.load(response)["total"] > 0, "synthetic inventory is empty"

    deadline = time.monotonic() + 20
    while True:
        result = policy_docs.retrieve("Which tags are required on Azure and GCP resources?", k=4)
        if (
            rag_down
            or result["status"] != "ok"
            or result["freshness"]["status"] == ("stale" if stale else "current")
        ):
            break
        assert time.monotonic() < deadline, "unexpected indexed source freshness"
        time.sleep(0.5)  # Provider caches freshness briefly; wait for its next real check.
    if rag_down:
        assert result["status"] in {"timeout", "unreachable"}, result["status"]
        print("PASS: RAG is unavailable; obs-agent health and inventory remain usable.")
        return

    assert result["status"] == "ok", result["status"]
    assert any(
        hit["source"] == "reference/tagging-standard.md" and hit["section"]
        for hit in result["hits"]
    ), "tagging-policy source/section missing"
    assert result["corpus"]["provenance"] == "available", "corpus provenance unavailable"
    assert result["corpus"]["corpus_sha256"], "corpus hash missing"
    assert result["freshness"]["status"] == ("stale" if stale else "current"), (
        "unexpected indexed source freshness"
    )
    empty = policy_docs.retrieve("zxqvnotinthesamplecorpus", max_distance=0.0)
    assert empty["status"] == "no_match", empty["status"]
    config = policy_docs.load_config()
    assert config.api_key, "RAG_API_KEY is required for the authentication check"
    with policy_docs.override(config=replace(config, api_key=config.api_key + "-wrong")):
        assert policy_docs.retrieve("Required tags?")["status"] == "unauthorized"
    print(
        "PASS: obs-agent API/inventory and real RAG retrieval, sources, provenance, no-match, auth."
    )


def break_contract(payload: dict) -> dict:
    payload.pop("hits")
    return payload


def contract_check() -> None:
    """Forward real retrieval over HTTP, changing only its contract at the boundary."""
    config = policy_docs.load_config()
    endpoint = policy_docs.validate(config)

    class IncompatibleResponse(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            request = Request(
                endpoint,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + config.api_key,
                },
            )
            with urlopen(request, timeout=config.timeout) as response:
                payload = break_contract(json.load(response))
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), IncompatibleResponse) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            with policy_docs.override(config=replace(config, url=url)):
                result = policy_docs.retrieve("Which tags are required on Azure and GCP resources?")
            assert result["status"] == "invalid_response", result["status"]
            assert not result["hits"], "incompatible hits were accepted"
            # Prove the unchanged normal smoke command fails, rather than blessing bad data.
            normal = subprocess.run(
                [sys.executable, __file__],
                env={**os.environ, "OBS_AGENT_RAG_URL": url},
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert normal.returncode != 0 and "AssertionError: invalid_response" in normal.stderr, (
                "normal smoke did not reject the incompatible HTTP response"
            )
            print("PASS: missing hits rejected; normal smoke fails; API/inventory stay usable.")
        finally:
            server.shutdown()
            thread.join(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-down", action="store_true", help="expect a stopped RAG container")
    parser.add_argument("--stale", action="store_true", help="expect changed source files")
    parser.add_argument("--contract-break", action="store_true")
    args = parser.parse_args()
    if args.contract_break:
        contract_check()
    else:
        check(args.rag_down, args.stale)
