"""Streamlit web UI for obs-agent.

Usage
-----
    uv run streamlit run app.py

Then open the URL shown in the terminal (typically http://localhost:8501).

You can pass the provider and model as query parameters:
    http://localhost:8501/?provider=ollama&model=qwen2.5:7b

Environment variables:
    DEEPSEEK_API_KEY   — required for DeepSeek (--provider deepseek)
    ANTHROPIC_API_KEY  — required for Anthropic
    OPENAI_API_KEY     — required for OpenAI
    OLLAMA_HOST        — optional, defaults to http://localhost:11434
"""

import os

import streamlit as st

from agent import (
    PROVIDERS,
    ConfigError,
    check_config,
    default_model,
    model_warning,
    run,
    served_by,
    stream_run,
)
from label_review import demo_enabled

# ── Page config ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="obs-agent — Multi-Cloud Cost Triage",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────

try:
    configured_provider, _ = check_config()
except ConfigError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()

st.sidebar.title("☁️ obs-agent")
st.sidebar.caption("Multi-Cloud Cost Triage Agent")

DEMO = demo_enabled()

if DEMO:
    # Hosted demo: no LLM, so no provider controls or key checks to confuse visitors.
    provider, effective_model, use_streaming = configured_provider, "", False
    st.sidebar.info("Hosted demo on synthetic data. Chat needs an LLM, so it is off here.")
    st.sidebar.page_link("pages/1_Label_review.py", label="Try Label review", icon="🏷️")
else:
    provider = st.sidebar.selectbox(
        "Provider",
        options=list(PROVIDERS),
        index=PROVIDERS.index(configured_provider),
        help="LLM provider to use for answering questions. The default comes from "
        "OBS_AGENT_PROVIDER (Ollama, local and free, when unset).",
    )

    model = st.sidebar.text_input(
        "Model (optional)",
        placeholder=f"default: {default_model(provider)}",
        help="Leave blank for the provider default (OBS_AGENT_MODEL for the default provider).",
    )
    effective_model = model or default_model(provider)
    if warning := model_warning(provider, effective_model):
        st.sidebar.warning(warning, icon="⚠️")

    use_streaming = st.sidebar.checkbox(
        "Stream responses",
        value=True,
        help="Show tokens as they arrive instead of waiting for the full answer.",
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Example questions:**")
    st.sidebar.markdown("- Which Azure subscription has the highest overage cost?")
    st.sidebar.markdown("- Show me the top 3 GCP projects")
    st.sidebar.markdown("- Compare costs across Azure and GCP")
    st.sidebar.markdown("- Any cost spikes above 200%?")
    st.sidebar.markdown("- What's the daily trend for sub-a1b2c3d4?")
    st.sidebar.markdown("- Which projects are missing a team tag?")
    st.sidebar.markdown("- What team does svc-ingest belong to?")
    st.sidebar.caption("Review and export tag suggestions on the **Label review** page.")
    st.sidebar.page_link(
        "pages/2_Import_inventory.py", label="Import a sanitized inventory", icon="📥"
    )

st.sidebar.divider()
with st.sidebar.expander("ℹ️ About"):
    st.markdown(
        "Infers missing team / env / cost_center tags across **Azure** and **GCP** "
        "with a scikit-learn model for human review, and answers cost questions "
        "through a tool-calling LLM agent (Claude, OpenAI, DeepSeek or Ollama). "
        "Runs on synthetic data. "
        "[Source on GitHub](https://github.com/redbarron23/obs-agent)"
    )

# ── API key check ──────────────────────────────────────────────────────

if not DEMO:
    if provider == "ollama":
        st.sidebar.info("🟢 Ollama — runs locally, no API key needed")
    elif provider == "deepseek" and not os.environ.get("DEEPSEEK_API_KEY"):
        st.sidebar.error("❌ `DEEPSEEK_API_KEY` not set in environment.")
    elif provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        st.sidebar.error("❌ `OPENAI_API_KEY` not set in environment.")
    elif provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        st.sidebar.error("❌ `ANTHROPIC_API_KEY` not set in environment.")
    else:
        st.sidebar.success(f"✅ {provider.title()} API key detected")

# ── Main chat ──────────────────────────────────────────────────────────

st.title("☁️ Multi-Cloud Cost Triage")

if DEMO:
    st.info(
        "This hosted demo has no LLM configured, so chat is off. The **Label review** "
        "page needs no LLM: it infers missing team/env tags with a scikit-learn model "
        "and lets you approve and export them.",
        icon="🧪",
    )
    st.page_link("pages/1_Label_review.py", label="Open Label review", icon="🏷️")
else:
    st.caption(
        f"Provider: **{provider}**  ·  Model: **{effective_model}**"
        + ("  ·  Streaming" if use_streaming else "")
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("via"):
            st.caption(msg["via"])

prompt = st.chat_input("Ask a question about cloud costs...", disabled=DEMO)
if DEMO:
    # `disabled` is enforced in the browser only; never reach the LLM from the demo.
    prompt = None
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            model_arg = model if model else None
            history = st.session_state.get("_agent_history")

            if use_streaming:
                holder: dict = {}
                st.write_stream(
                    stream_run(
                        prompt,
                        provider=provider,
                        model=model_arg,
                        verbose=True,
                        messages=history,
                        result_holder=holder,
                    )
                )
                answer = holder["answer"]
                st.session_state["_agent_history"] = holder["messages"]
            else:
                with st.spinner("Agent is thinking..."):
                    answer, updated = run(
                        prompt,
                        provider=provider,
                        model=model_arg,
                        verbose=True,
                        messages=history,
                    )
                st.markdown(answer)
                st.session_state["_agent_history"] = updated

            answered_by = served_by(st.session_state["_agent_history"]) or effective_model
            via = f"Answered by {provider} · {answered_by}"
            st.caption(via)
            st.session_state.messages.append({"role": "assistant", "content": answer, "via": via})
        except Exception as e:
            error_msg = f"Error: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
