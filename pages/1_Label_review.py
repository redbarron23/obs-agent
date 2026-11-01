"""Label review page: work through untagged / suspect tags and export the result.

Opened from the sidebar of the main app (`uv run streamlit run app.py`).
Decisions are shared with the REST API through the same SQLite database
(OBS_AGENT_DECISIONS, default label_decisions.db).
"""

import os
import tempfile
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

import policy_docs
import workflow_metrics
from label_config import inventory_config
from label_review import (
    MAX_NOTE,
    MAX_REVOCATION_REASON,
    DecisionError,
    DecisionStore,
    collaboration_state,
    coverage,
    current_decisions,
    demo_enabled,
    editor_changes,
    export,
    feedback_metrics,
    pending_changes,
    review_queue,
)
from labeling import AUTO_APPLY_THRESHOLD
from review_context import review_context
from review_workspaces import load_review

st.set_page_config(page_title="Label review — obs-agent", page_icon="🏷️", layout="wide")
workflow_metrics.bind(surface="ui")

DEMO = demo_enabled()
imported_inventory = None if DEMO else st.session_state.get("imported_inventory")
if DEMO:
    # Public demo: each browser session gets its own throwaway decisions file,
    # so visitors never see or overwrite each other's choices.
    st.session_state.setdefault("demo_id", uuid.uuid4().hex)
    demo_file = f"obs-agent-{st.session_state.demo_id}.db"
    store = DecisionStore(Path(tempfile.gettempdir()) / demo_file)
elif st.session_state.get("review_id"):
    imported_inventory, store = load_review(st.session_state.review_id)
    st.session_state["imported_inventory"] = imported_inventory
elif imported_inventory is not None:
    st.error("Save or reopen this inventory on the Import page before reviewing.")
    st.stop()
else:
    store = DecisionStore()
st.session_state.setdefault("editor_version", 0)


def _refresh():
    # A new editor key drops stale in-table edits after decisions change.
    st.session_state.editor_version += 1
    st.rerun()


# ── Header ─────────────────────────────────────────────────────────────

st.title("🏷️ Label review")
st.caption(
    "Suggested `team` / `env` tags for untagged Azure subscriptions and GCP projects, "
    f"plus existing tags the model thinks are wrong. Suggestions with a model score "
    f"≥{AUTO_APPLY_THRESHOLD:.0%} are eligible for expedited review; everything else needs an "
    "explicit human decision. The score ranks suggestions; it is not a calibrated probability. "
    "Where ownership evidence is missing, unsupported or conflicting, no team is suggested: "
    "the row says why, and only an explicit override resolves it."
)

if imported_inventory is not None:
    st.info(
        "Reviewing your imported inventory. Its snapshot, vocabulary, and decisions are kept in a "
        "saved local review database. Reopen it from Import / reopen inventory after a restart.",
        icon="📥",
    )
elif DEMO:
    st.info(
        "Demo on synthetic data: your decisions are private to this browser session "
        "and are discarded when it ends. Try accepting a few, then download the export.",
        icon="🧪",
    )

reviewer = st.text_input(
    "Reviewer name or ID",
    key="reviewer",
    value="demo-reviewer" if DEMO else os.environ.get("OBS_AGENT_REVIEWER", ""),
    max_chars=128,
    help="Recorded with each approval. This local identifier is self-reported.",
).strip()
if not reviewer:
    st.info("Enter your reviewer name or ID before saving or bulk-accepting decisions.")
context = review_context(imported_inventory)
with st.expander("Review snapshot"):
    st.code(f"Inventory: {context['inventory_id']}\nInference: {context['model_version']}")
valid_decisions = current_decisions(store, imported_inventory)
if len(valid_decisions) < len(store.all()):
    st.warning(
        "Earlier decisions lack matching snapshot/version evidence. Review them again to export."
    )
cov = coverage(imported_inventory)
decided = {f"{d['resource_id']}|{d['label']}": d for d in valid_decisions}
cols = st.columns(7)
cols[0].metric("Need attention", cov["needs_attention"], help=f"of {cov['total']} resources")
cols[1].metric("Untagged team", cov["labels"]["team"]["untagged"])
cols[2].metric("Untagged env", cov["labels"]["env"]["untagged"])
cols[3].metric("Suspect tags", sum(c["suspect"] for c in cov["labels"].values()))
cols[4].metric("Auto-apply ready", sum(c["auto_apply_ready"] for c in cov["labels"].values()))
cols[5].metric(
    "Insufficient evidence",
    sum(c["insufficient_evidence"] for c in cov["labels"].values()),
    help="Untagged labels with no suggestion: "
    + ", ".join(
        f"{reason.replace('_', ' ')} {n}"
        for reason, n in cov["labels"]["team"]["insufficient_evidence_reasons"].items()
    ),
)
cols[6].metric("Reviewed", len(decided))

quality = feedback_metrics(store.feedback())
with st.expander("Review quality", expanded=bool(quality["reviewed"])):
    st.caption(
        "Reviewer-agreement metrics compare each suggestion with the reviewer-resolved value. "
        "They are useful for monitoring, not an unbiased production benchmark."
    )
    qcols = st.columns(4)
    qcols[0].metric("Review actions", quality["reviewed"])
    qcols[1].metric("Scored", quality["scored"])
    qcols[2].metric("Unresolved", quality["unresolved"])
    qcols[3].metric("Expedited-review threshold", f"{quality['auto_apply_threshold']:.0%}")

    if not quality["scored"]:
        st.info(
            "No reviewer-resolved labels yet. Accept, override, or reject an existing tag "
            "to score it."
        )
    else:
        for label_name, summary in quality["labels"].items():
            mcols = st.columns(3)
            mcols[0].metric(f"{label_name} precision", f"{summary['precision']:.1%}")
            mcols[1].metric(f"{label_name} recall", f"{summary['recall']:.1%}")
            mcols[2].metric(f"{label_name} macro-F1", f"{summary['macro_f1']:.2f}")

        trend = [
            {
                "month": period["period"],
                "label": label_name,
                **{k: period[k] for k in ("scored", "precision", "recall", "macro_f1")},
            }
            for label_name, summary in quality["labels"].items()
            for period in summary["periods"]
        ]
        if trend:
            st.caption("Monthly reviewer-agreement trend")
            st.dataframe(pd.DataFrame(trend), hide_index=True, width="stretch")

        buckets = [
            {"label": label_name, **bucket}
            for label_name, summary in quality["labels"].items()
            for bucket in summary["confidence_buckets"]
        ]
        st.caption("Model-score buckets (the threshold is the high-score drift check)")
        st.dataframe(pd.DataFrame(buckets), hide_index=True, width="stretch")

# ── Filters ────────────────────────────────────────────────────────────

st.sidebar.header("Filters")
label = st.sidebar.radio("Label", ["team", "env"], horizontal=True)
status = st.sidebar.selectbox("Status", ["untagged or suspect", "untagged", "suspect"])
cloud = st.sidebar.selectbox("Cloud", ["both", "azure", "gcp"])
lo, hi = st.sidebar.slider(
    "Model score",
    0.0,
    1.0,
    (0.0, 1.0),
    step=0.05,
    help="Labels withheld for insufficient evidence have no score and count as 0.",
)
only_insufficient = st.sidebar.checkbox("Only insufficient evidence", value=False)
search = st.sidebar.text_input("Resource contains")
hide_reviewed = st.sidebar.checkbox("Hide reviewed", value=False)
assigned_to_me = st.sidebar.checkbox(
    "Assigned to me",
    value=False,
    disabled=not reviewer,
    help="Labels assigned to the reviewer name or ID above.",
)
only_unresolved = st.sidebar.checkbox("Only unresolved questions", value=False)
drift_filter = st.session_state.get("drift_filter")
drifted = None
if drift_filter and drift_filter["review_id"] == st.session_state.get("review_id"):
    if st.sidebar.checkbox(f"Only new or changed since {drift_filter['since']}", value=True):
        drifted = set(drift_filter["resource_ids"])

rows = review_queue(
    label=label,
    status=None if status.startswith("untagged or") else status,
    cloud=None if cloud == "both" else cloud,
    min_confidence=lo,
    max_confidence=hi,
    search=search or None,
    inventory=imported_inventory,
    evidence="insufficient" if only_insufficient else None,
    store=store,
    assignee=reviewer if assigned_to_me and reviewer else None,
    unresolved=True if only_unresolved else None,
)
# Collaboration metadata is shown alongside, never used to resolve or export a label.
collab = collaboration_state(store.collaboration_history())

records = []
for r in rows:
    lr = r["labels"][label]
    d = decided.get(f"{r['resource_id']}|{label}")
    item = collab.get((r["resource_id"], label)) or {}
    if hide_reviewed and d:
        continue
    if drifted is not None and r["resource_id"] not in drifted:
        continue
    records.append(
        {
            "resource_id": r["resource_id"],
            "cloud": r["cloud"],
            "parent": r["parent"],
            "owners": ", ".join(r["owners"]),
            "status": lr["status"],
            "current": lr["current"] or "",
            "suggested": lr["suggested"],
            "confidence": lr["confidence"],
            "action": lr["action"],
            "evidence": (lr["evidence"] or {}).get("status", "").replace("_", " "),
            "why": f"{lr['source']}: {lr['reason']}",
            "alternatives": ", ".join(
                f"{a['value']} {a['confidence']:.0%}" for a in lr["alternatives"]
            ),
            "cost_center": (
                "Conflicting: " + ", ".join(r["cost_center"]["conflicting"])
                if r["cost_center"]["conflicting"]
                else r["cost_center"]["implied"] or "No mapping"
            ),
            "decision": d["decision"] if d else "",
            "override": d["value"] if d and d["decision"] == "override" else None,
            "assignee": item.get("assignee") or "",
            "questions": item.get("unresolved_questions", 0),
        }
    )
df = pd.DataFrame(records)

# ── Review table ───────────────────────────────────────────────────────

st.subheader(f"{len(df)} resources to review for `{label}`")

if df.empty:
    st.success("Nothing matches these filters.")
else:
    edited = st.data_editor(
        df,
        key=f"editor-{label}-{st.session_state.editor_version}",
        hide_index=True,
        width="stretch",
        height=min(600, 38 + 35 * len(df)),
        disabled=[c for c in df.columns if c not in ("decision", "override")],
        # Editable columns sit next to the suggestion so no sideways scrolling.
        column_order=[
            "resource_id",
            "suggested",
            "confidence",
            "decision",
            "override",
            "assignee",
            "questions",
            "current",
            "status",
            "action",
            "evidence",
            "cloud",
            "parent",
            "owners",
            "alternatives",
            "cost_center",
            "why",
        ],
        column_config={
            "resource_id": st.column_config.TextColumn("Resource", pinned=True),
            "confidence": st.column_config.ProgressColumn(
                "Model score", min_value=0.0, max_value=1.0, format="percent"
            ),
            "why": st.column_config.TextColumn("Why", width="large"),
            "assignee": st.column_config.TextColumn("Assigned to"),
            "questions": st.column_config.NumberColumn(
                "Unresolved questions", help="Open or answered, not yet resolved."
            ),
            "decision": st.column_config.SelectboxColumn(
                "Decision",
                options=["", "accept", "reject", "override"],
                help="accept = apply the suggestion · reject = keep today's tag · "
                "override = apply the value in the next column (the only choice when no "
                "value is suggested)",
            ),
            "override": st.column_config.SelectboxColumn(
                "Override value",
                options=sorted(inventory_config(imported_inventory)[label]["values"]),
            ),
        },
    )

    rationale = st.text_input(
        "Rationale for these decisions (optional)",
        key="decision_rationale",
        max_chars=MAX_NOTE,
        help="Why you overrode (or accepted / rejected) a suggestion. Kept in the review history "
        "with each decision saved; never in plans, exports, logs or metrics.",
    ).strip()
    revoke_reason = st.text_input(
        "Reason for clearing a decision (optional)",
        key="revoke_reason",
        max_chars=MAX_REVOCATION_REASON,
        help="Clearing a decision revokes it: the reviewer, time and this reason are kept in "
        "the review history. The reason is never written to logs or metrics.",
    ).strip()
    left, right = st.columns([1, 3])
    if left.button("Save decisions", icon=":material/save:", type="primary", disabled=not reviewer):
        errors, saved = [], 0
        for rid, decision, override in editor_changes(df, edited):
            write = workflow_metrics.operation("decision_write", invalid=(DecisionError,))
            try:
                with write:
                    if decision:
                        store.set(
                            rid,
                            label,
                            decision,
                            override,
                            imported_inventory,
                            reviewer=reviewer,
                            rationale=rationale or None,
                        )
                    elif not store.revoke(
                        rid, label, reviewer=reviewer, reason=revoke_reason or None
                    ):
                        errors.append(f"{rid}: no decision to clear; it was already removed.")
                        continue
                saved += 1
            except DecisionError as exc:
                errors.append(f"{rid}: {exc}")
            except Exception:
                errors.append(
                    f"{rid}: the decision could not be saved. Reference: `{write.correlation_id}`"
                )
        for e in errors:
            st.error(e)
        if saved and not errors:
            st.toast(f"Saved {saved} decision(s)")
            _refresh()

    auto = df[(df["action"] == "auto-apply") & (df["decision"] == "")]
    if right.button(
        f"Accept all {len(auto)} auto-apply suggestions shown",
        icon=":material/done_all:",
        disabled=auto.empty or not reviewer,
    ):
        bulk_errors = []
        for rid in auto["resource_id"]:
            write = workflow_metrics.operation("decision_write", invalid=(DecisionError,))
            try:
                with write:
                    store.set(rid, label, "accept", inventory=imported_inventory, reviewer=reviewer)
            except DecisionError as exc:
                bulk_errors.append(f"{rid}: {exc}")
            except Exception:
                bulk_errors.append(
                    f"{rid}: the decision could not be saved. Reference: `{write.correlation_id}`"
                )
        for e in bulk_errors:
            st.error(e)
        if not bulk_errors:
            st.toast(f"Accepted {len(auto)} suggestion(s)")
            _refresh()

# ── Collaboration ──────────────────────────────────────────────────────

if not df.empty:
    st.subheader("Assignments, notes and questions")
    st.caption(
        "Record who should investigate a label, short notes, and unresolved questions. "
        "None of this is an approval: only a decision above changes the export. "
        "Text is kept in the review history, never in logs, metrics, plans or exports."
    )
    target = st.selectbox("Resource", list(df["resource_id"]), key="collab_resource")
    state = store.collaboration_item(target, label)

    def _collaborate(action, success: str) -> None:
        if not reviewer:
            st.error("Enter your reviewer name or ID first.")
            return
        write = workflow_metrics.operation("collaboration_write", invalid=(DecisionError, KeyError))
        try:
            with write:
                result = action()
        except (DecisionError, KeyError) as exc:
            st.error(str(exc) if isinstance(exc, DecisionError) else "Unknown resource.")
            return
        except Exception:
            st.error(f"Could not save. Reference: `{write.correlation_id}`")
            return
        if result is None:
            st.info("Nothing changed.")
            return
        st.toast(success)
        _refresh()

    acol, ncol = st.columns(2)
    acol.markdown(
        f"**Assigned to:** {state['assignee']} (by {state['assigned_by']}, {state['assigned_at']})"
        if state["assignee"]
        else "**Assigned to:** nobody"
    )
    assignee = acol.text_input("Assign to reviewer ID", key="collab_assignee", max_chars=128)
    b1, b2 = acol.columns(2)
    if b1.button("Assign", key="collab_assign", disabled=not reviewer or not assignee.strip()):
        _collaborate(
            lambda: store.assign(target, label, assignee, imported_inventory, reviewer=reviewer),
            "Assigned",
        )
    if b2.button("Unassign", key="collab_unassign", disabled=not reviewer or not state["assignee"]):
        _collaborate(lambda: store.unassign(target, label, reviewer=reviewer), "Unassigned")

    note = ncol.text_area("Note or question", key="collab_text", max_chars=MAX_NOTE, height=80)
    b3, b4 = ncol.columns(2)
    if b3.button("Add note", key="collab_note", disabled=not reviewer or not note.strip()):
        _collaborate(
            lambda: store.add_note(target, label, note, imported_inventory, reviewer=reviewer),
            "Note added",
        )
    if b4.button("Ask question", key="collab_ask", disabled=not reviewer or not note.strip()):
        _collaborate(
            lambda: store.ask_question(target, label, note, imported_inventory, reviewer=reviewer),
            "Question opened",
        )

    unresolved = {q["id"]: q for q in state["questions"] if q["status"] != "resolved"}
    if unresolved:
        question = unresolved[
            st.selectbox(
                "Unresolved question",
                list(unresolved),
                format_func=lambda i: f"#{i} ({unresolved[i]['status']}) {unresolved[i]['text']}",
                key="collab_question",
            )
        ]
        reply = st.text_input("Answer or resolution", key="collab_reply", max_chars=MAX_NOTE)
        b5, b6 = st.columns([1, 3])
        if b5.button("Answer", key="collab_answer", disabled=not reviewer or not reply.strip()):
            _collaborate(
                lambda: store.answer_question(question["id"], reply, reviewer=reviewer),
                "Answer added",
            )
        if b6.button("Resolve", key="collab_resolve", disabled=not reviewer):
            _collaborate(
                lambda: store.resolve_question(question["id"], reply or None, reviewer=reviewer),
                "Question resolved",
            )
    if state["history"]:
        with st.expander(f"History ({len(state['history'])} events)"):
            st.dataframe(
                pd.DataFrame(state["history"])[
                    ["created_at", "event", "reviewer", "assignee", "question_id", "text"]
                ],
                hide_index=True,
                width="stretch",
            )

# ── Policy context (optional) ──────────────────────────────────────────


def _plain(value: str) -> str:
    """Service-supplied text for a Markdown code span: no backticks or line breaks."""
    return " ".join(str(value).replace("`", "'").split()) or "-"


def _show_excerpts(hits: list[dict]) -> None:
    for h in hits:
        distance = "unknown" if h["distance"] is None else f"{h['distance']:.3f}"
        st.markdown(
            f"**{h['rank']}.** `{_plain(h['source'])}` § `{_plain(h['section'] or '(no heading)')}`"
            f" · distance {distance}"
        )
        # Plain text, never Markdown or HTML: retrieved documents are untrusted.
        st.text(h["text"])


def _show_policy_context(shown: dict, facts: dict) -> None:
    result = shown["result"]
    st.markdown("**Resource facts and model score** (this snapshot's inference)")
    st.caption("Policy excerpts below never change these values or establish an owner.")
    st.dataframe(pd.DataFrame([facts]), hide_index=True, width="stretch")
    st.markdown("**Policy evidence** (retrieved documentation, untrusted)")
    st.caption(f"Query sent: {shown['question']}")
    status = result["status"]
    if status not in policy_docs.ANSWERED:
        show = st.error if status == "misconfigured" else st.warning
        show(
            f"Policy context unavailable ({status.replace('_', ' ')}): {result['message']} "
            "Review, decisions and exports are unaffected.",
            icon=":material/cloud_off:",
        )
        return
    st.caption(policy_docs.corpus_line(result))
    if warning := policy_docs.freshness_warning(result):
        st.warning(warning, icon=":material/history:")
    if status == "no_match":
        st.info("No documentation section matched this label. The corpus has no supporting text.")
    else:
        if conflicts := policy_docs.possible_conflicts(result["hits"]):
            st.warning(
                "Unresolved: these sections came from more than one document and may conflict: "
                + ", ".join(conflicts)
                + ". obs-agent does not decide between them."
            )
        _show_excerpts(result["hits"])
    st.markdown("**Not evaluated by obs-agent**")
    st.caption(policy_docs.EVALUATED_FIELDS)
    if shown["not_evaluated"]:
        _show_excerpts(shown["not_evaluated"])
    else:
        st.info(
            f"The corpus returned no '{policy_docs.NOT_EVALUATED_SECTION}' section, so the "
            "out-of-scope requirements cannot be cited here."
        )


if not df.empty:
    st.subheader("Policy context")
    rag = policy_docs.load_config()
    if not rag.enabled:
        st.caption(
            "Optional: set `OBS_AGENT_RAG_URL` to a local rag-observability retrieval API to show "
            "cited tagging-policy excerpts beside a label. Review works the same without it."
        )
    else:
        snapshot = (
            f"imported review {st.session_state['review_id']}"
            if imported_inventory is not None
            else "synthetic demo inventory"
        )
        st.caption(
            f"Snapshot: {snapshot} (inventory sha256:{context['inventory_id'][:12]}…) · "
            f"Documentation service: {policy_docs.describe(rag)}. Only a fixed question about "
            f"the `{label}` tag is sent; no resource name, owner, parent or tag value."
        )
        policy_target = st.selectbox("Resource", list(df["resource_id"]), key="policy_resource")
        row = df[df["resource_id"] == policy_target].iloc[0]
        policy_key = [
            context["inventory_id"],
            st.session_state.get("review_id"),
            policy_target,
            label,
        ]
        if st.button("Show policy context", icon=":material/menu_book:", key="policy_show"):
            st.session_state["policy_context"] = {
                "key": policy_key,
                **policy_docs.review_policy_context(label, row["status"]),
            }
        shown = st.session_state.get("policy_context")
        # Never show context fetched for another resource, label or snapshot.
        if shown and shown["key"] == policy_key:
            _show_policy_context(
                shown,
                {
                    "resource_id": policy_target,
                    "label": label,
                    "status": row["status"],
                    "current": row["current"],
                    "suggested": row["suggested"],
                    "model score": row["confidence"],
                    "evidence": row["evidence"],
                },
            )

# ── Export ─────────────────────────────────────────────────────────────

st.divider()
st.subheader("Export")
include_auto = st.checkbox(
    f"Include unreviewed auto-apply suggestions (≥{AUTO_APPLY_THRESHOLD:.0%})",
    value=False,
    help="Off by default: export only what a reviewer accepted or overrode.",
)
# One "export" operation per render: the page prepares every download up front.
preparing = workflow_metrics.operation("export", invalid=(DecisionError,))
try:
    with preparing:
        changes = pending_changes(store, include_auto, imported_inventory)
        reviewed_changes = pending_changes(store, include_auto=False, inventory=imported_inventory)
        downloads = {
            "plan": export(store, "plan", inventory=imported_inventory),
            **{
                fmt: export(store, fmt, include_auto, imported_inventory)
                for fmt in ("csv", "sh", "tfvars")
            },
        }
except DecisionError as exc:
    st.error(str(exc))
    st.stop()
except Exception:
    st.error(
        "The export could not be prepared. Ask the operator to check the server log. "
        f"Reference: `{preparing.correlation_id}`"
    )
    st.stop()
st.caption(
    f"{len(changes)} resource(s) would change. Overrides win over accepts, "
    "accepts over auto-apply; rejected suggestions are never exported."
)

c1, c2, c3, c4 = st.columns(4)
c1.download_button(
    "Reviewed change plan",
    downloads["plan"],
    "approved-label-change-plan.json",
    "application/json",
    help="Reviewed decisions only, with the before/after diff and approval evidence.",
    icon=":material/fact_check:",
    disabled=not reviewed_changes,
)
c2.download_button(
    "CSV",
    downloads["csv"],
    "label-changes.csv",
    "text/csv",
    icon=":material/download:",
    disabled=not changes,
)
c3.download_button(
    "az / gcloud script",
    downloads["sh"],
    "apply-labels.sh",
    "text/x-shellscript",
    icon=":material/download:",
    disabled=not changes,
)
c4.download_button(
    "Terraform tfvars",
    downloads["tfvars"],
    "labels.auto.tfvars.json",
    "application/json",
    icon=":material/download:",
    disabled=not changes,
)

if changes:
    with st.expander("Preview changes"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "resource_id": c["resource_id"],
                        "cloud": c["cloud"],
                        "label": k,
                        "previous": c["previous"][k],
                        "new": v,
                        "how": how,
                    }
                    for c in changes
                    for k, (v, how) in c["tags"].items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
