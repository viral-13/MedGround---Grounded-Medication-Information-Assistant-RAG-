#!/usr/bin/env python3
"""
app.py - Step 6: a minimal Streamlit UI for the grounded medication-info RAG.

This is a THIN presentation layer only. All retrieval, generation, and refusal
logic lives in rag.py and is reused as-is via query(); nothing is reimplemented
here.

Run:
    pip install -r requirements.txt
    streamlit run app.py        # opens at http://localhost:8501
"""

import streamlit as st

import rag   # reuse query() + config; do not duplicate any logic

EXAMPLES = [
    "Can I take warfarin with aspirin?",
    "What is metformin used for?",
    "What are the side effects of sertraline?",
    "What is the recommended dose of Ozempic?",
    "What's the capital of France?",
]


@st.cache_resource(show_spinner="Loading embedding model + FAISS index ...")
def warm_rag():
    """Load the embedding model + FAISS index ONCE per process. rag.py loads
    these lazily and caches them in a module global; we just trigger that load
    here so it happens a single time rather than on every interaction."""
    rag._resources()
    return True


def set_question(text):
    st.session_state["question_input"] = text


def render_result(res):
    refused = res["refused"]
    top = res["top_score"]
    thr = rag.RETRIEVAL_THRESHOLD

    # --- status badge -------------------------------------------------------
    if refused:
        color, label = "#b45309", "REFUSED"      # amber
    else:
        color, label = "#15803d", "ANSWERED"     # green
    st.markdown(
        f"<span style='background:{color};color:white;padding:3px 12px;"
        f"border-radius:12px;font-weight:600;font-size:0.85rem'>{label}</span>",
        unsafe_allow_html=True,
    )
    st.write("")

    # --- answer -------------------------------------------------------------
    st.markdown(res["answer"])

    # --- sources ------------------------------------------------------------
    if res["citations"]:
        st.markdown("#### Sources")
        for c in res["citations"]:
            st.markdown(f"- [{c['drug']} - {c['section']}]({c['source_url']})")

    # --- how it decided -----------------------------------------------------
    with st.expander("How it decided"):
        gate = "above" if top >= thr else "below"
        passed = "passed" if top >= thr else "blocked"
        st.markdown(
            f"**Layer 1 - retrieval gate:** top match **{top:.2f}** is {gate} the "
            f"**{thr:.2f}** gate -> {passed}."
        )
        if refused and top < thr:
            st.markdown("**Outcome:** refused at **Layer 1** "
                        "(nothing relevant retrieved - no LLM call made).")
        elif refused:
            st.markdown("**Layer 2 - grounded generation:** passed the gate, but the "
                        "model could not ground an answer in the retrieved text, so it "
                        "**self-refused** (`can_answer = false`).")
        else:
            st.markdown("**Layer 2 - grounded generation:** answered using only the "
                        "retrieved chunks, with citations.")

        st.markdown("**Retrieved chunks (drug - section - score):**")
        for r in res["retrieved"]:
            st.markdown(f"- {r['drug']} - {r['section']} - `{r['score']:.3f}`")


def main():
    st.set_page_config(page_title="MedGround", page_icon="💊")
    warm_rag()

    st.title("MedGround - Grounded Medication-Information Assistant")
    st.write(
        "Answers questions about 50 common drugs using only public FDA drug-label "
        "text, cites every source, and refuses when it can't ground an answer."
    )
    st.caption(
        "Demo only - not medical advice. Built on public FDA (openFDA / DailyMed) "
        "label data."
    )

    st.markdown("**Try an example:**")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        col.button(ex, on_click=set_question, args=(ex,), use_container_width=True)

    st.session_state.setdefault("question_input", "")
    question = st.text_input("Your question", key="question_input",
                             placeholder="e.g. Can I take warfarin with aspirin?")
    ask = st.button("Ask", type="primary")

    if ask:
        q = (question or "").strip()
        if not q:
            st.info("Please enter a question (or pick an example above).")
            return
        try:
            with st.spinner("Retrieving and grounding ..."):
                res = rag.query(q)
        except Exception as e:                       # noqa: BLE001
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                st.error("The language model is rate-limited right now (free-tier "
                         "quota). Please wait a moment and try again.")
            elif "503" in msg or "UNAVAILABLE" in msg:
                st.error("The language model is temporarily overloaded. Please try "
                         "again in a few seconds.")
            else:
                st.error("Something went wrong while answering. Please try again.")
            return
        render_result(res)


if __name__ == "__main__":
    main()
