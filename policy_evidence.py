"""
Policy evidence a reviewer chose to retain with a label decision (#103).

When a reviewer has looked at the policy excerpts beside a label (docs/policy-context.md) and
saved a decision, they can attach the excerpts they consulted to that decision. Each attached
excerpt becomes one row of the append-only `policy_evidence` table in the same SQLite file as the
decision (`DecisionStore.attach_policy_evidence`). A row records:

- what was consulted: source, section, chunk ID, rank, distance, the excerpt's SHA-256 and (by
  default) its exact text, plus the fixed question that retrieved it and when;
- which revision of the documentation it came from: the corpus identity (`corpus_sha256`,
  `ingested_at`, embedding model, collection, document count), its provenance and freshness,
  and the per-document policy metadata (policy ID, version, effective dates, supersession) when
  the retrieval service reports it (rag-observability contract 1.1);
- which decision it supported: the decision's `review_feedback` row, decision, value, reviewer
  and time, and its inventory fingerprint and inference version.

Attaching evidence is not an approval, never changes a decision, label confidence, plan, export
or cloud tag, and cannot establish an owner. Excerpt text stays in the private decision database
(and its backups); it is never written to exports, plans, logs or workflow metrics.

`inspection_reasons()` compares retained evidence with the decision's current state and, when
given one, a fresh retrieval for the same question, and lists why a reviewer may want to look at
the decision again (changed or superseded policy, changed corpus, revoked decision, unavailable
provenance or text). It only reports; nothing is modified.
"""

from __future__ import annotations

import hashlib
import json

import policy_docs

# Policy applicability statuses (rag-observability contract 1.1) that mean the excerpt should not
# be read as current, applicable guidance.
NOT_APPLICABLE = ("superseded", "expired", "conflict", "invalid", "future")

# Columns of the policy_evidence table, in order. Created by DecisionStore._initialise().
COLUMNS = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "resource_id": "TEXT NOT NULL",
    "label": "TEXT NOT NULL",
    # The decision this evidence was consulted for.
    "feedback_id": "INTEGER",
    "decision": "TEXT NOT NULL",
    "decision_value": "TEXT",
    "decision_reviewer": "TEXT",
    "decided_at": "TEXT NOT NULL",
    "inventory_id": "TEXT",
    "model_version": "TEXT",
    # Who attached it, and when.
    "reviewer": "TEXT NOT NULL",
    "attached_at": "TEXT NOT NULL",
    # The retrieval.
    "question": "TEXT NOT NULL",
    "retrieval_status": "TEXT NOT NULL",
    "retrieved_at": "TEXT",
    # The excerpt.
    "rank": "INTEGER",
    "chunk_id": "TEXT",
    "source": "TEXT",
    "section": "TEXT",
    "distance": "REAL",
    "excerpt_sha256": "TEXT NOT NULL",
    "excerpt_text": "TEXT",
    # The corpus revision it came from.
    "corpus_provenance": "TEXT",
    "corpus_sha256": "TEXT",
    "ingested_at": "TEXT",
    "embedding_model": "TEXT",
    "collection": "TEXT",
    "document_count": "INTEGER",
    "freshness_status": "TEXT",
    "freshness_checked_at": "TEXT",
    # Per-document policy metadata as JSON, or NULL when the service did not report it.
    "policy_status": "TEXT",
    "policy": "TEXT",
}


class EvidenceError(ValueError):
    """Evidence that cannot be attached (no decision, nothing selected, no answered retrieval)."""


def excerpt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _policy(hit: dict) -> dict | None:
    policy = hit.get("policy")
    return policy if isinstance(policy, dict) else None


def evidence_rows(
    result: dict,
    selected: list[dict],
    *,
    question: str,
    retrieved_at: str | None,
    retain_text: bool = True,
) -> list[dict]:
    """The excerpt and corpus columns for each *selected* hit of one retrieval *result*.

    Raises EvidenceError unless the retrieval was answered and every selected hit is one of its
    hits, so evidence is never recorded against a corpus it did not come from.
    """
    if result.get("status") not in policy_docs.ANSWERED or not result.get("hits"):
        raise EvidenceError("Only excerpts from an answered retrieval can be attached.")
    if not selected:
        raise EvidenceError("Select at least one excerpt to attach.")
    hits = result["hits"]
    if any(hit not in hits for hit in selected):
        raise EvidenceError("Selected excerpts must come from the retrieval shown.")
    corpus = result.get("corpus") or {}
    freshness = result.get("freshness") or {}
    rows = []
    for hit in selected:
        policy = _policy(hit)
        document_count = corpus.get("document_count")
        rows.append(
            {
                "question": question,
                "retrieval_status": result["status"],
                "retrieved_at": retrieved_at,
                "rank": hit.get("rank"),
                "chunk_id": hit.get("chunk_id") or None,
                "source": hit.get("source") or None,
                "section": hit.get("section") or None,
                "distance": hit.get("distance"),
                "excerpt_sha256": excerpt_sha256(hit["text"]),
                "excerpt_text": hit["text"] if retain_text else None,
                "corpus_provenance": corpus.get("provenance"),
                "corpus_sha256": corpus.get("corpus_sha256"),
                "ingested_at": corpus.get("ingested_at"),
                "embedding_model": corpus.get("embedding_model"),
                "collection": corpus.get("collection"),
                "document_count": document_count if isinstance(document_count, int) else None,
                "freshness_status": freshness.get("status"),
                "freshness_checked_at": freshness.get("checked_at"),
                "policy_status": policy.get("status") if policy else None,
                "policy": json.dumps(policy, sort_keys=True) if policy else None,
            }
        )
    return rows


def _identity(row: dict) -> str:
    policy = json.loads(row["policy"]) if row.get("policy") else {}
    if policy.get("policy_id"):
        return f"policy {policy['policy_id']}@{policy.get('version') or 'unversioned'}"
    return "this excerpt"


def inspection_reasons(
    row: dict, active_feedback_id: int | None, current: dict | None = None
) -> list[str]:
    """Why a reviewer may want to look at the decision this evidence row supported again.

    *active_feedback_id* is the `review_feedback` row of the label's current decision (None when
    there is none). *current* is an optional fresh `policy_docs.retrieve()` result; it is only
    compared when it answered the same question. Nothing is modified.
    """
    reasons = []
    if row["feedback_id"] is None or row["feedback_id"] != active_feedback_id:
        reasons.append(
            "The decision this evidence supported has since been revoked or replaced; "
            "the evidence is kept as history."
        )
    if row["corpus_provenance"] != "available" or not row["corpus_sha256"]:
        reasons.append(
            "Corpus provenance was unavailable when this was attached, so the policy revision "
            "cannot be identified."
        )
    if row["excerpt_text"] is None:
        reasons.append(
            "Only the excerpt's SHA-256 was retained; the exact historical text is unavailable."
        )
    if row["policy_status"] in NOT_APPLICABLE:
        reasons.append(f"When attached, {_identity(row)} was {row['policy_status']}.")
    if (
        current is None
        or current.get("status") not in policy_docs.ANSWERED
        or current.get("question") != row["question"]
    ):
        return reasons
    now = (current.get("corpus") or {}).get("corpus_sha256")
    if row["corpus_sha256"] and now and now != row["corpus_sha256"]:
        reasons.append(
            f"The documentation index has changed since this was attached "
            f"(corpus {row['corpus_sha256'][:12]} → {now[:12]})."
        )
    match = next(
        (
            hit
            for hit in current["hits"]
            if (row["chunk_id"] and hit.get("chunk_id") == row["chunk_id"])
            or (
                not row["chunk_id"]
                and (hit.get("source"), hit.get("section")) == (row["source"], row["section"])
            )
        ),
        None,
    )
    if match is None:
        reasons.append(
            "The current index did not return this excerpt for the same question; it may have "
            "been removed or moved."
        )
        return reasons
    if excerpt_sha256(match["text"]) != row["excerpt_sha256"]:
        reasons.append("The excerpt's text has changed in the current index.")
    status = (_policy(match) or {}).get("status")
    if status in NOT_APPLICABLE and status != row["policy_status"]:
        reasons.append(f"The policy is now {status} in the current index.")
    return reasons
