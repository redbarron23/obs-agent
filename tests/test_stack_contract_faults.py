"""The controlled fault changes only the required HTTP response field."""

import json
from pathlib import Path

import policy_docs
from scripts.local_stack_smoke import break_contract


def test_missing_hits_is_rejected_by_production_client():
    payload = json.loads(
        (Path(__file__).parents[1] / "fixtures/rag_retrieve_tagging_standard.json").read_text()
    )
    expected = {key: value for key, value in payload.items() if key != "hits"}
    changed = break_contract(payload)
    assert changed == expected
    config = policy_docs.RagConfig("http://127.0.0.1:8765")
    with policy_docs.override(config=config, transport=policy_docs.stub_transport(200, changed)):
        result = policy_docs.retrieve("Required tags?")
    assert result["status"] == "invalid_response"
    assert result["hits"] == []
