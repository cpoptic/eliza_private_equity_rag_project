"""
SEC Filing Intelligence — Streamlit frontend.

Run:  streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SEC Filing Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — dark financial terminal aesthetic
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #21262d;
    --accent:    #f0b429;
    --accent2:   #58a6ff;
    --green:     #3fb950;
    --red:       #f85149;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'IBM Plex Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Header */
.site-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    padding: 8px 0 24px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
}
.site-header h1 {
    font-family: var(--mono);
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--accent);
    margin: 0;
    letter-spacing: -0.02em;
}
.site-header .sub {
    font-size: 0.8rem;
    color: var(--muted);
    font-family: var(--mono);
}

/* Index status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: var(--mono);
    font-size: 0.72rem;
    font-weight: 600;
}
.status-badge.ready   { background: rgba(63,185,80,0.15);  color: var(--green); border: 1px solid rgba(63,185,80,0.3); }
.status-badge.building{ background: rgba(240,180,41,0.15); color: var(--accent); border: 1px solid rgba(240,180,41,0.3); }
.status-badge.empty   { background: rgba(248,81,73,0.12);  color: var(--red);   border: 1px solid rgba(248,81,73,0.25); }

/* Answer box */
.answer-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 20px 24px;
    margin-top: 16px;
    font-size: 0.93rem;
    line-height: 1.7;
}

/* Chunk cards */
.chunk-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 12px 14px;
    margin-bottom: 8px;
    font-size: 0.82rem;
}
.chunk-header {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--accent2);
    margin-bottom: 6px;
    font-weight: 600;
}
.chunk-score {
    display: inline-block;
    background: rgba(88,166,255,0.1);
    color: var(--accent2);
    border-radius: 3px;
    padding: 1px 6px;
    font-family: var(--mono);
    font-size: 0.7rem;
    margin-left: 8px;
}
.chunk-method {
    display: inline-block;
    background: rgba(240,180,41,0.1);
    color: var(--accent);
    border-radius: 3px;
    padding: 1px 6px;
    font-family: var(--mono);
    font-size: 0.7rem;
    margin-left: 4px;
}
.chunk-text {
    color: var(--muted);
    line-height: 1.5;
    font-size: 0.8rem;
}

/* Metric tiles */
.metric-row { display: flex; gap: 12px; margin-bottom: 16px; }
.metric-tile {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 16px;
    text-align: center;
}
.metric-tile .val {
    font-family: var(--mono);
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--accent);
}
.metric-tile .lbl {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 2px;
}

/* Example questions */
.example-q {
    background: rgba(88,166,255,0.06);
    border: 1px solid rgba(88,166,255,0.15);
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 0.83rem;
    color: var(--text);
    cursor: pointer;
    margin-bottom: 6px;
    font-family: var(--sans);
    transition: border-color 0.2s;
}

/* Streamlit widget overrides */
.stTextArea textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
    font-size: 0.9rem !important;
    border-radius: 6px !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(240,180,41,0.2) !important;
}
.stButton button {
    background: var(--accent) !important;
    color: #0d1117 !important;
    font-family: var(--mono) !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 4px !important;
    font-size: 0.85rem !important;
}
.stButton button:hover {
    background: #f5c842 !important;
}
div[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
.stSelectbox select, .stSelectbox > div {
    background: var(--surface) !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Pipeline (cached across sessions)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_pipeline():
    from src.pipeline import build_pipeline
    return build_pipeline()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    "What are the primary risk factors facing Apple, Tesla, and NVIDIA, and how do they compare?",
    "How has NVIDIA's revenue and growth outlook changed over the last two years?",
    "What regulatory risks do major pharmaceutical companies face and how are they addressing them?",
    "Compare Microsoft and Google's cloud business strategy and revenue growth.",
    "What cybersecurity risks are disclosed by major tech companies in their most recent filings?",
    "How has Apple's supply chain risk evolved across its last three annual reports?",
]

with st.sidebar:
    st.markdown("""
    <div style='font-family:IBM Plex Mono,monospace;font-size:1rem;font-weight:600;
                color:#f0b429;padding-bottom:12px;border-bottom:1px solid #21262d;'>
        ⚙ Configuration
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Index management
    corpus_dir = st.text_input("Corpus directory", value="data/edgar_corpus")

    col1, col2 = st.columns(2)
    with col1:
        build_btn = st.button("Build Index", use_container_width=True)
    with col2:
        rebuild_btn = st.button("Rebuild", use_container_width=True)

    st.markdown("---")
    st.markdown("**Example Questions**")
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if st.button(q[:60] + "…" if len(q) > 60 else q, key=f"eq_{i}", use_container_width=True):
            st.session_state["prefill_question"] = q

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#8b949e;font-family:IBM Plex Mono,monospace;'>"
        f"Model: {os.getenv('LLM_MODEL','claude-sonnet-4-5')}<br>"
        f"Embedder: {os.getenv('EMBEDDER','openai')}<br>"
        f"Store: {os.getenv('VECTOR_STORE','chroma')}"
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Index build actions
# ---------------------------------------------------------------------------

pipeline = get_pipeline()

if build_btn or rebuild_btn:
    force = rebuild_btn
    with st.spinner("Building index…"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(current, total, message):
            if total > 0:
                progress_bar.progress(current / total)
            status_text.text(message)

        result = pipeline.index_corpus(
            corpus_dir=corpus_dir,
            progress_callback=update_progress,
            force_reindex=force,
        )
    progress_bar.empty()
    status_text.empty()
    if result["status"] == "skipped":
        st.info(result["reason"])
    else:
        st.success(
            f"✓ Indexed {result['chunks_indexed']} chunks from "
            f"{result['files_processed']} files in {result['elapsed_ms']/1000:.1f}s"
        )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown("""
<div class="site-header">
    <h1>SEC // FILING INTELLIGENCE</h1>
    <span class="sub">RAG · Single-Call · Financial Analysis</span>
</div>
""", unsafe_allow_html=True)

# Index status + stats
chunk_count = pipeline._store.count()
status_class = "ready" if chunk_count > 0 else "empty"
status_label = f"● {chunk_count:,} chunks indexed" if chunk_count > 0 else "● No index — build first"

st.markdown(f"""
<div class="metric-row">
  <div class="metric-tile"><div class="val">{chunk_count:,}</div><div class="lbl">Indexed Chunks</div></div>
  <div class="metric-tile"><div class="val">{os.getenv('LLM_MODEL','claude-sonnet-4-5').split('/')[-1]}</div><div class="lbl">LLM Model</div></div>
  <div class="metric-tile"><div class="val">{os.getenv('EMBEDDER','openai').upper()}</div><div class="lbl">Embedder</div></div>
  <div class="metric-tile"><div class="val">{os.getenv('VECTOR_STORE','chroma').upper()}</div><div class="lbl">Vector Store</div></div>
</div>
<span class="status-badge {status_class}">{status_label}</span>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Question input
prefill = st.session_state.pop("prefill_question", "")
question = st.text_area(
    "Business question",
    value=prefill,
    height=90,
    placeholder="e.g. What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?",
    label_visibility="collapsed",
)

run_col, _ = st.columns([1, 4])
with run_col:
    run_btn = st.button("▶  Analyze", use_container_width=True)

# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

if run_btn and question.strip():
    if chunk_count == 0:
        st.error("Index is empty. Build the index first using the sidebar.")
    else:
        with st.spinner("Retrieving and analyzing…"):
            result = pipeline.query(question.strip())

        # Parse info
        ctx = result.query_context
        st.markdown(f"""
        <div style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#8b949e;
                    margin-bottom:12px;'>
            Query type: <span style='color:#f0b429'>{ctx.query_type}</span>
            &nbsp;·&nbsp; Tickers: <span style='color:#58a6ff'>{', '.join(ctx.tickers) or 'none detected'}</span>
            &nbsp;·&nbsp; Chunks used: <span style='color:#58a6ff'>{len(result.chunks)}</span>
            &nbsp;·&nbsp; Latency: <span style='color:#3fb950'>{result.latency_ms:.0f}ms</span>
        </div>
        """, unsafe_allow_html=True)

        # Answer
        st.markdown(
            f'<div class="answer-container">{result.answer}</div>',
            unsafe_allow_html=True,
        )

        # Retrieved context expander (transparency panel)
        chunks_display = [
            {
                "header": rc.chunk.provenance_header(),
                "score": f"{rc.score:.3f}",
                "method": rc.retrieval_method,
                "text_preview": rc.chunk.text[:300].replace("\n", " "),
            }
            for rc in result.chunks
        ]
        with st.expander(f"📎 Retrieved Context — {len(result.chunks)} chunks", expanded=False):
            for chunk_info in chunks_display:
                st.markdown(f"""
                <div class="chunk-card">
                    <div class="chunk-header">
                        {chunk_info['header']}
                        <span class="chunk-score">score: {chunk_info['score']}</span>
                        <span class="chunk-method">{chunk_info['method']}</span>
                    </div>
                    <div class="chunk-text">{chunk_info['text_preview']}</div>
                </div>
                """, unsafe_allow_html=True)

elif run_btn:
    st.warning("Please enter a question.")
