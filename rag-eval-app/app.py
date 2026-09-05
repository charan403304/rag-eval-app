"""
Streamlit front-end for the RAG app.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import CONFIG  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

st.set_page_config(page_title="Docs RAG Assistant", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Building index...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline.from_corpus()


def main():
    st.title("📚 Internal Docs Assistant")
    st.caption(
        f"Hybrid (BM25 + vector) retrieval over a small internal-docs corpus, with citations "
        f"and an explicit refusal guardrail. Embeddings: `{CONFIG.embedding_provider}` · "
        f"LLM: `{CONFIG.llm_provider}`"
    )

    with st.sidebar:
        st.header("About")
        st.markdown(
            "This app answers questions **only** from the markdown docs in `data/corpus/` "
            "(deployment, API guidelines, incident response, DB migrations). It will refuse "
            "to answer anything outside that corpus rather than guess."
        )
        st.markdown("---")
        st.markdown(
            "**Provider config** is set via environment variables — see `.env.example`. "
            "Currently running with the offline defaults (`hash` embeddings / `mock` LLM) "
            "unless you've set `EMBEDDING_PROVIDER` / `LLM_PROVIDER`."
        )
        st.markdown("---")
        if st.button("Run evaluation suite"):
            st.session_state["run_eval"] = True

    pipeline = get_pipeline()

    query = st.text_input(
        "Ask a question about deployment, the API, incidents, or DB migrations:",
        placeholder="e.g. What happens when the API rate limit is exceeded?",
    )

    if query:
        with st.spinner("Retrieving and generating..."):
            response = pipeline.answer(query)

        if response.refused:
            st.warning(response.answer)
        else:
            st.markdown("### Answer")
            st.markdown(response.answer)

        with st.expander(f"Retrieved context ({len(response.retrieved)} chunks)", expanded=False):
            for i, rc in enumerate(response.retrieved, start=1):
                cited = "✅ cited" if i in response.citations_used else "— not cited"
                st.markdown(
                    f"**[{i}] {rc.chunk.doc_id} / {rc.chunk.section}** "
                    f"(fused score: {rc.score:.4f}, {cited})"
                )
                st.text(rc.chunk.text)
                st.markdown("---")

        with st.expander("Query stats (cost / latency)", expanded=False):
            st.json(response.stats.as_dict())

    if st.session_state.get("run_eval"):
        st.markdown("---")
        st.subheader("Evaluation Report")
        with st.spinner("Running eval suite..."):
            from eval.run_eval import run
            summary = run(use_llm_judge=False)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Refusal accuracy", f"{summary['refusal_accuracy']:.0%}")
        col2.metric("Retrieval recall", f"{summary.get('retrieval_recall_avg', 0):.0%}")
        col3.metric("Citation validity", f"{summary['citation_validity_avg']:.0%}")
        col4.metric("Keyword recall", f"{summary.get('keyword_recall_avg', 0):.0%}")

        st.dataframe(
            [{k: v for k, v in r.items() if k != "answer"} for r in summary["rows"]],
            use_container_width=True,
        )
        st.session_state["run_eval"] = False


if __name__ == "__main__":
    main()
