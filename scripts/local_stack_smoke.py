"""Check the running local Compose stack without an LLM or cloud credentials."""

import argparse
import json
import os
from dataclasses import replace
from urllib.request import Request, urlopen

import policy_docs


def check(rag_down: bool = False) -> None:
    with urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
        assert response.status == 200, "obs-agent API is not healthy"
    request = Request(
        "http://127.0.0.1:8000/labels/coverage",
        headers={"X-API-Key": os.environ["OBS_AGENT_API_KEY"]},
    )
    with urlopen(request, timeout=30) as response:
        assert json.load(response)["total"] > 0, "synthetic inventory is empty"

    result = policy_docs.retrieve("Which tags are required on Azure and GCP resources?", k=4)
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
    assert result["freshness"]["status"] == "current", "indexed source files are not current"
    empty = policy_docs.retrieve("zxqvnotinthesamplecorpus", max_distance=0.0)
    assert empty["status"] == "no_match", empty["status"]
    config = policy_docs.load_config()
    assert config.api_key, "RAG_API_KEY is required for the authentication check"
    with policy_docs.override(config=replace(config, api_key=config.api_key + "-wrong")):
        assert policy_docs.retrieve("Required tags?")["status"] == "unauthorized"
    print(
        "PASS: obs-agent API/inventory and real RAG retrieval, sources, provenance, no-match, auth."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-down", action="store_true", help="expect a stopped RAG container")
    check(parser.parse_args().rag_down)
