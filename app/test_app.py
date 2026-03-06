from __future__ import annotations

import os
import time
import queue
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer

from utils.db import run_query
from utils.images import find_image_path
from utils.rag import (
    to_vec_literal,
    aggregate_docs_from_hits,
    select_chunks_for_top_docs,
    build_prompt_and_source_map,
)
from utils.upload import save_uploaded_image

try:
    from google import genai
except Exception:
    genai = None


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DEFAULT_IMAGE_BASE_DIR   = os.environ.get("IMAGE_BASE_DIR",       "/Users/shaldonng/Desktop/Y4S1/FYP/app/centralised_document_images")
DEFAULT_TABLE_NAME       = os.environ.get("TABLE_NAME",           "suki_fyp_vectors_v2")
DEFAULT_CHUNK_TABLE_NAME = os.environ.get("CHUNK_TABLE_NAME",     "suki_fyp_chunks_v1")
DEFAULT_LLM_MODEL        = os.environ.get("LLM_MODEL",            "gemini-2.5-flash")
DEFAULT_EMBEDDING_MODEL  = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
DEFAULT_UPLOAD_DIR       = os.environ.get("UPLOAD_DIR",           "/Users/shaldonng/Desktop/Y4S1/FYP/app/uploads")
os.makedirs(DEFAULT_UPLOAD_DIR, exist_ok=True)

# Embedding modes — grouped into the two core modes for the toggle
LAYOUT_EMBEDDINGS = {
    "Layout-Only (Visual)": "layout_embedding",
    "LayoutLMv3":           "layoutlmv3_embedding",
    "ColPali Embeddings":   "colpali_embeddings",
}
SEMANTIC_EMBEDDINGS = {
    "Text-Only (Semantic)": "text_embedding",
    "Combined (Hybrid)":    "combined_embedding",
}
ALL_EMBEDDING_COLUMNS: Dict[str, str] = {**LAYOUT_EMBEDDINGS, **SEMANTIC_EMBEDDINGS}

DISTANCE_METRICS: Dict[str, Dict[str, Any]] = {
    "Cosine Distance":           {"operator": "<=>", "description": "1 − cosine similarity. Best general-purpose metric for normalised embeddings.", "binary_only": False},
    "L2 (Euclidean)":            {"operator": "<->", "description": "Standard Euclidean distance. Sensitive to vector magnitude.",                  "binary_only": False},
    "Inner Product (Negative)":  {"operator": "<#>", "description": "Negative inner product. Use for un-normalised embeddings.",                    "binary_only": False},
    "L1 (Manhattan)":            {"operator": "<+>", "description": "Sum of absolute differences. More robust to outliers.",                        "binary_only": False},
    "Hamming Distance":          {"operator": "<~>", "description": "Bit-level difference. Binary vectors only.",                                   "binary_only": True},
    "Jaccard Distance":          {"operator": "<%>", "description": "Set overlap distance. Binary vectors only.",                                   "binary_only": True},
}


@dataclass(frozen=True)
class AppConfig:
    image_base_dir:       str
    table_name:           str
    chunk_table_name:     str
    llm_model:            str
    embedding_model_name: str

cfg = AppConfig(
    image_base_dir       = DEFAULT_IMAGE_BASE_DIR,
    table_name           = DEFAULT_TABLE_NAME,
    chunk_table_name     = DEFAULT_CHUNK_TABLE_NAME,
    llm_model            = DEFAULT_LLM_MODEL,
    embedding_model_name = DEFAULT_EMBEDDING_MODEL,
)


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="DocLens",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #FAFAF8; }

/* Keep header so sidebar toggle arrow remains visible */
#MainMenu  { visibility: hidden; }
footer     { visibility: hidden; }
header     { visibility: visible !important; background: transparent !important; box-shadow: none !important; }
[data-testid="stToolbar"]      { visibility: hidden; }
[data-testid="stDecoration"]   { display: none; }
[data-testid="stStatusWidget"] { display: none; }
[data-testid="collapsedControl"],
button[kind="header"] {
    display: flex !important; visibility: visible !important;
    opacity: 1 !important; pointer-events: auto !important;
}

[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E8E6E0; }

[data-testid="stMetric"] { background:#FFFFFF; border:1px solid #E8E6E0; border-radius:12px; padding:16px !important; }
[data-testid="stMetricLabel"] { font-size:12px !important; color:#6B6860 !important; }
[data-testid="stMetricValue"] { font-family:'DM Mono',monospace !important; font-size:24px !important; font-weight:700 !important; }

/* Primary buttons */
.stButton > button {
    background-color: #2563EB; color: white; border: none; border-radius: 8px;
    padding: 9px 22px; font-family: 'DM Sans', sans-serif; font-weight: 500; font-size: 13px;
    transition: opacity 0.15s;
}
.stButton > button:hover { opacity: 0.85; background-color: #2563EB !important; color: white !important; }

/* Secondary button override via container class */
div[data-testid="stButton"].secondary > button,
button[kind="secondary"] {
    background-color: #FFFFFF !important; color: #1A1916 !important;
    border: 1px solid #E8E6E0 !important;
}

/* Inputs */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stTextArea textarea {
    background: #FFFFFF; border: 1px solid #E8E6E0 !important;
    border-radius: 8px; font-family: 'DM Sans', sans-serif; font-size: 13px; color: #1A1916;
}
.stTextInput > div > div > input:focus, .stTextArea textarea:focus {
    border-color: #BFCFED !important; box-shadow: 0 0 0 2px rgba(37,99,235,0.08) !important;
}

/* Progress */
.stProgress > div > div { background-color: #2563EB !important; border-radius: 99px; }
.stProgress > div       { background-color: #E8E6E0; border-radius: 99px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF; border: 1px solid #E8E6E0; border-radius: 10px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] { border-radius: 7px; font-size: 13px; font-weight: 500; color: #6B6860; padding: 8px 18px; }
.stTabs [aria-selected="true"] { background: #2563EB !important; color: white !important; }

[data-testid="stDataFrame"] { border: 1px solid #E8E6E0; border-radius: 10px; overflow: hidden; }
.streamlit-expanderHeader { font-size: 12px; font-weight: 600; color: #1A1916; background: #FFFFFF; border: 1px solid #E8E6E0; border-radius: 8px; }
hr { border-color: #E8E6E0 !important; }

/* Shared utility classes */
.card { background: #FFFFFF; border: 1px solid #E8E6E0; border-radius: 14px; padding: 20px 22px; margin-bottom: 10px; }
.section-label { font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #9B9890; display: block; margin-bottom: 12px; }
.doc-thumb { background: #EEF3FD; border: 1px solid #BFCFED; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; font-family: 'DM Mono', monospace; font-size: 11px; font-weight: 600; color: #2563EB; width: 40px; height: 40px; flex-shrink: 0; }

.tag { display: inline-block; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 500; }
.tag-blue  { background: #EEF3FD; color: #2563EB; border: 1px solid #BFCFED; }
.tag-green { background: #ECFDF5; color: #16A34A; border: 1px solid #86efac44; }
.tag-amber { background: #FFFBEB; color: #D97706; border: 1px solid #fcd34d44; }
.tag-gray  { background: #F4F4F0; color: #6B6860; border: 1px solid #E8E6E0; }
.tag-purple{ background: #F5F3FF; color: #7C3AED; border: 1px solid #c4b5fd44; }

/* Mode toggle cards */
.mode-card {
    border: 2px solid #E8E6E0; border-radius: 14px; padding: 18px 20px;
    cursor: pointer; transition: all .2s; background: #FFFFFF;
}
.mode-card.active-layout  { border-color: #7C3AED; background: #F5F3FF; }
.mode-card.active-semantic { border-color: #2563EB; background: #EEF3FD; }
.mode-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); }

/* Chat bubbles */
.user-bubble { background: #1A1916; color: #fff; border-radius: 14px 4px 14px 14px; padding: 12px 16px; font-size: 13px; line-height: 1.65; max-width: 72%; margin-left: auto; }
.assistant-bubble { background: #FFFFFF; color: #1A1916; border: 1px solid #E8E6E0; border-radius: 4px 14px 14px 14px; padding: 12px 16px; font-size: 13px; line-height: 1.65; max-width: 72%; }
.source-tag { display: inline-block; background: #FAFAF8; border: 1px solid #E8E6E0; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-family: 'DM Mono', monospace; color: #6B6860; margin: 2px 3px; }

/* Score bar */
.score-row { display:flex; align-items:center; gap:14px; background:#FFFFFF; border:1px solid #E8E6E0; border-radius:12px; padding:12px 16px; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CACHED MODELS
# ─────────────────────────────────────────────

@st.cache_resource
def load_embedder():
    return SentenceTransformer(cfg.embedding_model_name)

@st.cache_resource
def load_gemini():
    if genai is None:
        return None
    key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or st.secrets.get("GEMINI_API_KEY", None)
    )
    return genai.Client(api_key=key) if key else None

# ── Pipeline embedder models — loaded once, reused across uploads ────────────
# Each returns the instantiated embedder class (with model already loaded),
# or None if the import/model load fails. The background thread receives these
# pre-loaded objects so it never triggers a slow model load mid-run.

@st.cache_resource
def load_custom_embedder(image_folder: str):
    try:
        from embedders.CustomEmbedder import CustomEmbedder
        return CustomEmbedder(image_folder=image_folder, num_workers=1)
    except Exception as e:
        return None

@st.cache_resource
def load_layoutlmv3_embedder(image_folder: str):
    try:
        from embedders.LayoutlmEmbedder import Layoutlmv3Embedder
        return Layoutlmv3Embedder(image_folder=image_folder, num_workers=1)
    except Exception as e:
        return None

embedder      = load_embedder()
gemini_client = load_gemini()

def embed_query_vec(text: str) -> List[float]:
    arr = embedder.encode([text], normalize_embeddings=True)
    v0  = arr[0]
    return v0.tolist() if hasattr(v0, "tolist") else list(v0)

def gemini_answer(prompt: str, model: str) -> str:
    if gemini_client is None:
        return "⚠ Gemini client not available — check google-genai install and GEMINI_API_KEY."
    resp = gemini_client.models.generate_content(model=model, contents=prompt)
    return (resp.text or "").strip()


# ─────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────

_defaults: Dict[str, Any] = {
    "page":              "Dashboard",
    "chat_history":      [],
    # similarity
    "sim_emb_label":     "Layout-Only (Visual)",
    "sim_metric_label":  "Cosine Distance",
    "sim_topk":          5,
    "sim_results":       None,
    "sim_cols":          None,
    "sim_anchor_uuid":   None,
    "sim_selected_doc":  None,
    # rag
    "rag_answer":        "",
    "rag_hits":          [],
    "rag_doc_rank":      pd.DataFrame(),
    "rag_source_map":    [],
    "rag_selected_doc":  None,
    "rag_last_query":    "",
    # upload
    "upload_stage":       "idle",
    "upload_file_name":   None,
    "upload_saved_list":  [],
    "pipeline_log":       [],
    "pipeline_done":      False,
    "pipeline_success":   False,
    "pipeline_queue":     None,
    "pipeline_progress":  0,
    # upload Q&A
    "upload_qa_query":    "",
    "upload_qa_answer":   None,
    "upload_qa_hits":     [],
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────
# SIDEBAR  (RAG config + corpus stats only)
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:4px 0 20px">
        <div style="width:34px;height:34px;background:#2563EB;border-radius:9px;
                    display:flex;align-items:center;justify-content:center;font-size:18px;color:white">◈</div>
        <div>
            <div style="font-size:15px;font-weight:700;color:#1A1916;letter-spacing:-0.01em">DocLens</div>
            <div style="font-size:10px;color:#9B9890;font-family:'DM Mono',monospace">pgvector · Gemini</div>
        </div>
    </div>
    <hr style="margin-bottom:16px">
    """, unsafe_allow_html=True)

    st.markdown("<span class='section-label'>RAG Configuration</span>", unsafe_allow_html=True)
    rag_chunk_table  = st.text_input("Chunks Table",        value=cfg.chunk_table_name)
    rag_topk_chunks  = st.slider("Retrieve Top-K Chunks",   5, 80, 30)
    rag_topk_docs    = st.slider("Show Top-K Documents",    1, 15,  5)
    rag_llm_model    = st.text_input("Gemini Model",        value=cfg.llm_model)
    rag_show_sources = st.checkbox("Show retrieved chunks", value=True)

    if genai is None:
        st.warning("Install SDK: `pip install google-genai`")
    elif gemini_client is None:
        st.warning("Set GEMINI_API_KEY env var or st.secrets.")

    st.markdown("<hr style='margin:16px 0'>", unsafe_allow_html=True)

    # Live corpus stats
    _r, _ = run_query(f'SELECT COUNT(DISTINCT image_uuid) FROM "{cfg.table_name}";')
    _doc_count = _r[0][0] if _r else "—"
    _rc, _ = run_query(f'SELECT COUNT(*) FROM "{cfg.chunk_table_name}";')
    _chunk_count = _rc[0][0] if _rc else "—"

    st.markdown(f"""
    <div style="background:#FAFAF8;border:1px solid #E8E6E0;border-radius:10px;padding:14px 16px">
        <div style="font-size:11px;color:#9B9890;margin-bottom:8px;font-weight:700;letter-spacing:0.07em;text-transform:uppercase">Corpus</div>
        <div style="font-size:20px;font-weight:700;color:#1A1916;font-family:'DM Mono',monospace">{_doc_count}</div>
        <div style="font-size:12px;color:#6B6860;margin-bottom:10px">documents indexed</div>
        <div style="font-size:18px;font-weight:700;color:#1A1916;font-family:'DM Mono',monospace">{_chunk_count}</div>
        <div style="font-size:12px;color:#6B6860;margin-bottom:6px">text chunks</div>
        <div style="font-size:10px;color:#9B9890;font-family:'DM Mono',monospace;margin-top:6px">{cfg.table_name}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def score_color(s: float) -> str:
    return "#16A34A" if s <= 0.25 else ("#2563EB" if s <= 0.50 else "#D97706")

def score_tag_cls(s: float) -> str:
    return "tag-green" if s <= 0.25 else ("tag-blue" if s <= 0.50 else "tag-amber")

def show_image(path: str, caption: str = "", **kwargs):
    """
    Safe st.image wrapper — opens via PIL so Streamlit never tries to
    serve the file through its media file handler (which causes
    MediaFileStorageError for absolute paths on disk).
    """
    try:
        img = Image.open(path)
        st.image(img, caption=caption, **kwargs)
    except Exception as e:
        st.warning(f"Could not load image: {e}")

def fetch_all_uuids() -> List[str]:
    rows, _ = run_query(f'SELECT DISTINCT image_uuid FROM "{cfg.table_name}" ORDER BY image_uuid ASC;')
    return [r[0] for r in rows] if rows else []

def rag_retrieve_chunks(query: str, chunk_table: str, k: int) -> List[Tuple]:
    qlit = to_vec_literal(embed_query_vec(query))
    sql  = f"""
        SELECT chunk_id, image_uuid, chunk_text,
               (chunk_embedding <=> %s::vector) AS cosine_distance
        FROM   {chunk_table}
        WHERE  chunk_embedding IS NOT NULL
        ORDER  BY chunk_embedding <=> %s::vector
        LIMIT  %s;
    """
    rows, _ = run_query(sql, (qlit, qlit, k))
    return rows or []

def run_similarity_query(anchor_uuid: str, col: str, op: str, k: int):
    sql = f"""
        SELECT t2.image_uuid,
               t2."{col}" {op} t1."{col}" AS score
        FROM   "{cfg.table_name}" t1,
               "{cfg.table_name}" t2
        WHERE  t1.image_uuid = %s
          AND  t1.image_uuid <> t2.image_uuid
          AND  t1."{col}" IS NOT NULL
          AND  t2."{col}" IS NOT NULL
        ORDER  BY score ASC
        LIMIT  {k};
    """
    return run_query(sql, (anchor_uuid,))


# ─────────────────────────────────────────────
# TOP NAV BAR
# ─────────────────────────────────────────────

nav_items = {
    "Dashboard":  "⊞  Dashboard",
    "Library":    "⊟  Library",
    "Similarity": "◈  Similarity Search",
    "RAG":        "◎  Text Q&A",
    "Upload":     "⊕  Upload",
}
_nav_cols = st.columns(len(nav_items))
for _col, (_key, _label) in zip(_nav_cols, nav_items.items()):
    with _col:
        _active = st.session_state.page == _key
        if st.button(_label, key=f"nav_{_key}", use_container_width=True,
                     type="primary" if _active else "secondary"):
            st.session_state.page = _key
            st.rerun()

# ── Global pipeline status banner — visible on every page ────────────────────
# Drains the queue here so it works regardless of which page the user is on.
_pipeline_stage = st.session_state.get("upload_stage", "idle")

if _pipeline_stage in ("processing", "done"):
    # Always drain the queue so log + progress stay up to date
    _log_q: queue.Queue = st.session_state.get("pipeline_queue")
    if _log_q:
        while True:
            try:
                _item = _log_q.get_nowait()
                if _item[0] == "__DONE__":
                    st.session_state.pipeline_done    = True
                    st.session_state.pipeline_success = _item[1]
                    st.session_state.upload_stage     = "done"
                elif _item[0] == "__PROGRESS__":
                    st.session_state.pipeline_progress = _item[1]
                elif _item[0] == "__ERROR__":
                    st.session_state.pipeline_log.append(("error", _item[1], "error"))
                else:
                    st.session_state.pipeline_log.append(_item)
            except queue.Empty:
                break

    _n_done  = st.session_state.get("pipeline_progress", 0)
    _n_total = 3
    _fname   = st.session_state.get("upload_file_name", "")

    if _pipeline_stage == "processing":
        _b1, _b2 = st.columns([6, 1])
        with _b1:
            st.progress(
                _n_done / _n_total,
                text=f"⚙️ Pipeline running — stage {_n_done}/{_n_total} complete · **{_fname}**  *(you can freely navigate while this runs)*"
            )
        with _b2:
            if st.button("↻ Refresh", key="banner_refresh"):
                st.rerun()
    elif _pipeline_stage == "done":
        _success = st.session_state.get("pipeline_success", False)
        if _success:
            st.success(f"✓ Pipeline complete for **{_fname}** — results saved to `embeddings/`")
        else:
            st.warning(f"⚠ Pipeline finished with errors for **{_fname}** — go to Upload to see the log.")
        _bc1, _bc2 = st.columns([1, 5])
        with _bc1:
            if st.button("Dismiss", key="banner_dismiss"):
                st.session_state.upload_stage = "idle"
                st.rerun()

st.markdown("<hr style='margin:10px 0 28px'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────

def page_dashboard():
    st.markdown("""
    <div style="margin-bottom:28px">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
            <span style="font-size:11px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
                         color:#2563EB;background:#EEF3FD;border:1px solid #BFCFED;
                         border-radius:99px;padding:3px 12px">FYP Demo</span>
            <span style="font-size:11px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
                         color:#7C3AED;background:#F5F3FF;border:1px solid #c4b5fd44;
                         border-radius:99px;padding:3px 12px">pgvector · Gemini</span>
        </div>
        <h1 style="font-size:32px;font-weight:700;letter-spacing:-0.03em;color:#1A1916;
                   line-height:1.2;margin:0 0 10px">
            Document Similarity<br>
            <span style="color:#2563EB">Search</span> &amp;
            <span style="color:#7C3AED">Retrieval</span> System
        </h1>
        <p style="font-size:14px;color:#6B6860;max-width:560px;line-height:1.7;margin:0">
            Explore how different vector representations affect document retrieval —
            switch between layout and semantic embeddings, query with natural language,
            and inspect source citations grounded in your corpus.
        </p>
    </div>
    """, unsafe_allow_html=True)

    r_docs,   _ = run_query(f'SELECT COUNT(DISTINCT image_uuid) FROM "{cfg.table_name}";')
    r_chunks, _ = run_query(f'SELECT COUNT(*) FROM "{cfg.chunk_table_name}";')
    n_docs   = r_docs[0][0]   if r_docs   else "—"
    n_chunks = r_chunks[0][0] if r_chunks else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents Indexed",  str(n_docs),                 "in corpus")
    c2.metric("Text Chunks",        str(n_chunks),               "indexed passages")
    c3.metric("Embedding Models",   str(len(ALL_EMBEDDING_COLUMNS)), "available")
    c4.metric("Distance Metrics",   str(len(DISTANCE_METRICS)),  "available")

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    st.markdown("<span class='section-label'>What would you like to do?</span>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
                <div style="width:40px;height:40px;background:#F5F3FF;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px;color:#7C3AED">◈</div>
                <span class="tag tag-purple">Track A · pgvector</span>
            </div>
            <div style="font-size:15px;font-weight:600;color:#1A1916;margin-bottom:8px">Image Similarity Search</div>
            <div style="font-size:13px;color:#6B6860;line-height:1.6">Toggle between <b>Layout mode</b> (find visually similar documents — same form type, vendor) and <b>Semantic mode</b> (find conceptually similar content regardless of appearance).</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open Similarity Search →", key="d_sim"):
            st.session_state.page = "Similarity"; st.rerun()

        st.markdown("""
        <div class="card" style="margin-top:4px">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
                <div style="width:40px;height:40px;background:#EEF3FD;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px;color:#2563EB">⊟</div>
                <span class="tag tag-gray">Library</span>
            </div>
            <div style="font-size:15px;font-weight:600;color:#1A1916;margin-bottom:8px">Document Library</div>
            <div style="font-size:13px;color:#6B6860;line-height:1.6">Browse all indexed documents and launch similarity search from any entry.</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Browse Library →", key="d_lib"):
            st.session_state.page = "Library"; st.rerun()

    with col2:
        st.markdown("""
        <div class="card">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
                <div style="width:40px;height:40px;background:#EEF3FD;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px;color:#2563EB">◎</div>
                <span class="tag tag-amber">Track B · RAG</span>
            </div>
            <div style="font-size:15px;font-weight:600;color:#1A1916;margin-bottom:8px">Text Q&A (RAG)</div>
            <div style="font-size:13px;color:#6B6860;line-height:1.6">Ask a natural language question — Gemini answers using chunks retrieved from pgvector with source citations.</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Open Text Q&A →", key="d_rag"):
            st.session_state.page = "RAG"; st.rerun()

        st.markdown("""
        <div class="card" style="margin-top:4px">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
                <div style="width:40px;height:40px;background:#EEF3FD;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px;color:#2563EB">⊕</div>
                <span class="tag tag-blue">Ingest</span>
            </div>
            <div style="font-size:15px;font-weight:600;color:#1A1916;margin-bottom:8px">Upload Document</div>
            <div style="font-size:13px;color:#6B6860;line-height:1.6">Save new document images to the upload directory for downstream embedding and indexing.</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Upload Document →", key="d_up"):
            st.session_state.page = "Upload"; st.rerun()


# ─────────────────────────────────────────────
# PAGE: DOCUMENT LIBRARY
# ─────────────────────────────────────────────

def page_library():
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown("<h1 style='font-size:22px;font-weight:700;letter-spacing:-0.02em;color:#1A1916;margin-bottom:4px'>Document Library</h1>", unsafe_allow_html=True)
    with h2:
        if st.button("⊕ Upload New", key="lib_up"):
            st.session_state.page = "Upload"; st.rerun()

    all_uuids = fetch_all_uuids()
    st.markdown(f"<p style='font-size:13px;color:#6B6860;margin-bottom:18px'>{len(all_uuids)} documents in <code>{cfg.table_name}</code></p>", unsafe_allow_html=True)

    search   = st.text_input("", placeholder="⌕  Filter by image UUID…", label_visibility="collapsed")
    filtered = [u for u in all_uuids if search.lower() in u.lower()]

    if not filtered:
        st.info("No documents match your filter, or the corpus is empty.")
        return

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, uuid in enumerate(filtered):
        with cols[i % 4]:
            img_path = find_image_path(uuid, cfg.image_base_dir, extra_dirs=[DEFAULT_UPLOAD_DIR])
            st.markdown(f"""
            <div class="card" style="padding:14px;text-align:center">
                <div class="doc-thumb" style="margin:0 auto 10px">{uuid[:4].upper()}</div>
                <div style="font-size:10px;font-weight:600;color:#1A1916;word-break:break-all;line-height:1.4">{uuid}</div>
            </div>""", unsafe_allow_html=True)
            if img_path:
                show_image(img_path, use_container_width=True)
            if st.button("Find Similar", key=f"lib_{i}"):
                st.session_state.sim_anchor_uuid = uuid
                st.session_state.sim_results     = None
                st.session_state.page            = "Similarity"
                st.rerun()


# ─────────────────────────────────────────────
# PAGE: SIMILARITY SEARCH  (core feature)
# ─────────────────────────────────────────────

def page_similarity():

    st.markdown("<h1 style='font-size:22px;font-weight:700;letter-spacing:-0.02em;color:#1A1916;margin-bottom:4px'>Image Similarity Search</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:13px;color:#6B6860;margin-bottom:28px'>Select an embedding representation and distance metric to explore how different vector representations affect document retrieval.</p>", unsafe_allow_html=True)

    all_uuids = fetch_all_uuids()
    if not all_uuids:
        st.warning("No documents found. Check the table name in the sidebar.")
        return

    # ── Embedding, Metric, Top-K ─────────────────────────────────────────────
    EMB_DESCRIPTIONS = {
        "Layout-Only (Visual)": ("🗂️", "Captures spatial structure, table alignment, and header positions. Best for finding documents of the same visual template or form type."),
        "LayoutLMv3":           ("📄", "Multimodal — encodes text content jointly with its physical position on the page. Understands both what is written and where."),
        "ColPali Embeddings":   ("🖼️", "Vision-language model that reads the full page as an image. Captures diagrams, logos, and visual-textual patterns together."),
        "Text-Only (Semantic)": ("💬", "Pure text meaning — ignores layout entirely. Best for finding documents that discuss the same topic, clause, or concept regardless of appearance."),
        "Combined (Hybrid)":    ("⚡", "Fuses layout and semantic signals into one vector. Balances visual structure and textual meaning simultaneously."),
    }

    all_emb_options    = list(ALL_EMBEDDING_COLUMNS.keys())
    all_metric_options = list(DISTANCE_METRICS.keys())

    if st.session_state.sim_emb_label not in all_emb_options:
        st.session_state.sim_emb_label = all_emb_options[0]
    if st.session_state.sim_metric_label not in DISTANCE_METRICS:
        st.session_state.sim_metric_label = "Cosine Distance"

    cfg_col1, cfg_col2, cfg_col3 = st.columns([2, 2, 1])

    with cfg_col1:
        emb_label = st.selectbox(
            "Embedding Representation",
            all_emb_options,
            index=all_emb_options.index(st.session_state.sim_emb_label),
            key="emb_select",
        )
        st.session_state.sim_emb_label = emb_label
        active_col    = ALL_EMBEDDING_COLUMNS[emb_label]
        is_layout_emb = emb_label in LAYOUT_EMBEDDINGS
        icon, desc    = EMB_DESCRIPTIONS[emb_label]
        # Description card under the dropdown
        with st.container(border=True):
            st.markdown(f"{icon} **{emb_label}**")
            st.caption(f"`{active_col}`")
            st.write(desc)

    with cfg_col2:
        metric_label = st.selectbox(
            "Distance Metric (pgvector)",
            all_metric_options,
            index=all_metric_options.index(st.session_state.sim_metric_label),
            key="metric_select",
        )
        st.session_state.sim_metric_label = metric_label
        m_info   = DISTANCE_METRICS[metric_label]
        operator = m_info["operator"]
        with st.container(border=True):
            st.markdown(f"**{metric_label}**")
            st.caption(f"Operator: `{operator}`")
            st.write(m_info["description"])
            if m_info["binary_only"]:
                st.warning("⚠ Requires binary vectors.")

    with cfg_col3:
        top_k = st.number_input("Top-K Results", min_value=1, max_value=20,
                                value=st.session_state.sim_topk, step=1)
        st.session_state.sim_topk = top_k

    st.divider()

    st.write("**Select Anchor Document & Run Search**")
    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        default_idx = all_uuids.index(st.session_state.sim_anchor_uuid) \
                      if st.session_state.sim_anchor_uuid in all_uuids else 0
        anchor_uuid = st.selectbox("Anchor Document", all_uuids, index=default_idx)

        if anchor_uuid != st.session_state.sim_anchor_uuid:
            st.session_state.sim_results     = None
            st.session_state.sim_cols        = None
            st.session_state.sim_selected_doc = None
        st.session_state.sim_anchor_uuid = anchor_uuid

        anchor_path = find_image_path(anchor_uuid, cfg.image_base_dir, extra_dirs=[DEFAULT_UPLOAD_DIR])
        if anchor_path:
            show_image(anchor_path, caption=anchor_uuid, use_container_width=True)
        else:
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:28px">
                <div class="doc-thumb" style="margin:0 auto 10px;width:52px;height:52px;font-size:14px">{anchor_uuid[:4].upper()}</div>
                <div style="font-size:12px;color:#9B9890">Image not found on disk</div>
            </div>""", unsafe_allow_html=True)

        # Config summary
        icon, _ = EMB_DESCRIPTIONS[emb_label]
        st.caption(f"{icon} **{emb_label}** · {metric_label} · Top-{top_k}")

    with right_col:
        if st.button(f"Run Search  {icon}  →", key="sim_run", use_container_width=True):
            with st.spinner(f"Querying pgvector using {emb_label}…"):
                results, cols = run_similarity_query(anchor_uuid, active_col, operator, top_k)
            st.session_state.sim_results     = results
            st.session_state.sim_cols        = cols
            if results:
                st.session_state.sim_selected_doc = results[0][0]

        # ── Results ──
        if st.session_state.sim_results:
            df = pd.DataFrame(st.session_state.sim_results, columns=st.session_state.sim_cols)

            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin:16px 0 12px">
                <span class="section-label" style="margin-bottom:0">Results</span>
                <span class="tag tag-blue">{emb_label}</span>
                <span class="tag tag-gray">{metric_label}</span>
            </div>""", unsafe_allow_html=True)

            for i, row in df.iterrows():
                uid   = row["image_uuid"]
                score = float(row["score"])
                bar_w = max(4, int((1 - min(score, 1.0)) * 100))
                st.markdown(f"""
                <div class="score-row">
                    <div style="font-size:12px;font-weight:700;color:#9B9890;font-family:'DM Mono',monospace;width:24px">#{i+1}</div>
                    <div class="doc-thumb" style="width:34px;height:34px;font-size:10px">{uid[:4].upper()}</div>
                    <div style="flex:1;min-width:0;font-size:12px;font-weight:600;color:#1A1916;word-break:break-all">{uid}</div>
                    <div style="display:flex;align-items:center;gap:10px;flex-shrink:0">
                        <div style="width:90px;height:6px;background:#F0EEE8;border-radius:99px;overflow:hidden">
                            <div style="width:{bar_w}%;height:100%;background:{score_color(score)};border-radius:99px"></div>
                        </div>
                        <span class="tag {score_tag_cls(score)}" style="font-family:'DM Mono',monospace;min-width:60px;text-align:center">{score:.4f}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

            similar_docs = df["image_uuid"].tolist()
            idx = similar_docs.index(st.session_state.sim_selected_doc) \
                  if st.session_state.sim_selected_doc in similar_docs else 0
            sel = st.selectbox("View this document →", similar_docs, index=idx, key="sim_doc_view")
            st.session_state.sim_selected_doc = sel

        elif st.session_state.sim_results is not None:
            st.warning("No results returned. The selected embedding column may be empty for this document.")
        else:
            st.markdown(f"""
            <div style="text-align:center;padding:60px 0;color:#9B9890">
                <div style="font-size:32px;margin-bottom:12px">{icon}</div>
                <div style="font-size:14px;font-weight:500">Select an anchor document and click Run Search</div>
                <div style="font-size:12px;margin-top:6px">Using <b>{emb_label}</b> · <b>{metric_label}</b></div>
            </div>""", unsafe_allow_html=True)

    # ── Side-by-side comparison ──────────────────────────────────────────────
    if st.session_state.sim_results and st.session_state.sim_selected_doc:
        st.divider()
        st.markdown("<span class='section-label'>Side-by-side Comparison</span>", unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.caption("🔵 Anchor Document")
            p = find_image_path(anchor_uuid, cfg.image_base_dir, extra_dirs=[DEFAULT_UPLOAD_DIR])
            if p: show_image(p, caption=anchor_uuid, use_container_width=True)
            else: st.warning("Anchor image not found on disk.")
        with c2:
            st.caption(f"{icon} Most Similar — {emb_label}")
            p = find_image_path(st.session_state.sim_selected_doc, cfg.image_base_dir, extra_dirs=[DEFAULT_UPLOAD_DIR])
            if p: show_image(p, caption=st.session_state.sim_selected_doc, use_container_width=True)
            else: st.warning("Similar image not found on disk.")

        # Per-embedding insight callout
        EMB_INSIGHTS = {
            "Layout-Only (Visual)":  "🗂️ **Layout-Only (Visual)** — results share similar spatial structure, table layouts, and header positions. Documents likely belong to the same form template or document type.",
            "LayoutLMv3":            "📄 **LayoutLMv3** — results are similar in both text content and physical layout position. This multimodal embedding captures where text appears on the page, not just what it says.",
            "ColPali Embeddings":    "🖼️ **ColPali** — results match at the full-page vision-language level. ColPali understands the page as an image, capturing visual patterns, diagrams, and text together.",
            "Text-Only (Semantic)":  "💬 **Text-Only (Semantic)** — results contain similar textual meaning. Documents may look completely different but share concepts, clauses, or business topics.",
            "Combined (Hybrid)":     "⚡ **Combined (Hybrid)** — results are similar across both layout and semantics. This fused embedding balances visual structure and textual meaning simultaneously.",
        }
        st.info(EMB_INSIGHTS.get(emb_label, "Results retrieved using the selected embedding and distance metric."))


# ─────────────────────────────────────────────
# PAGE: RAG Q&A
# ─────────────────────────────────────────────

def page_rag():
    st.markdown("<h1 style='font-size:22px;font-weight:700;letter-spacing:-0.02em;color:#1A1916;margin-bottom:4px'>Text Q&A</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:13px;color:#6B6860;margin-bottom:18px'>Ask questions — Gemini answers using chunks retrieved from pgvector with inline citations.</p>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#FFFFFF;border:1px solid #E8E6E0;border-radius:10px;padding:10px 18px;margin-bottom:20px;
                display:flex;gap:24px;align-items:center;font-size:12px;color:#6B6860;flex-wrap:wrap">
        <span>Embedder: <b style="color:#1A1916">{cfg.embedding_model_name}</b></span>
        <span style="color:#E8E6E0">|</span>
        <span>LLM: <b style="color:#1A1916">{rag_llm_model}</b></span>
        <span style="color:#E8E6E0">|</span>
        <span>Top-K chunks: <b style="color:#1A1916">{rag_topk_chunks}</b></span>
        <span style="color:#E8E6E0">|</span>
        <span>Top-K docs: <b style="color:#1A1916">{rag_topk_docs}</b></span>
        <span style="color:#E8E6E0">|</span>
        <span>Table: <b style="color:#1A1916">{rag_chunk_table}</b></span>
    </div>""", unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div style="display:flex;justify-content:flex-end;margin-bottom:16px"><div class="user-bubble">{msg["text"]}</div></div>', unsafe_allow_html=True)
        else:
            src_html = ""
            if msg.get("sources"):
                tags = "".join(f'<span class="source-tag">{s}</span>' for s in msg["sources"])
                src_html = f'<div style="margin-top:8px"><span style="font-size:11px;color:#9B9890">Sources: </span>{tags}</div>'
            st.markdown(f"""
            <div style="display:flex;gap:12px;margin-bottom:16px;align-items:flex-start">
                <div style="width:32px;height:32px;border-radius:50%;background:#EEF3FD;border:1px solid #BFCFED;
                            display:flex;align-items:center;justify-content:center;font-size:13px;color:#2563EB;font-weight:600;flex-shrink:0">◈</div>
                <div><div class="assistant-bubble">{msg["text"]}</div>{src_html}</div>
            </div>""", unsafe_allow_html=True)

    with st.form("rag_form", clear_on_submit=True):
        c_in, c_btn = st.columns([6, 1])
        with c_in:
            user_query = st.text_input("", placeholder="e.g. Show me all lease agreements that involve a security deposit…", label_visibility="collapsed")
        with c_btn:
            submitted = st.form_submit_button("Send →")

    if submitted and user_query.strip():
        q = user_query.strip()
        st.session_state.chat_history.append({"role": "user", "text": q})

        with st.spinner("Retrieving chunks from pgvector…"):
            hits = rag_retrieve_chunks(q, rag_chunk_table.strip(), rag_topk_chunks)

        if not hits:
            st.session_state.chat_history.append({
                "role": "assistant",
                "text": "⚠ No chunks retrieved. Check the chunks table name and that embeddings are populated.",
                "sources": [],
            })
        else:
            agg_df        = aggregate_docs_from_hits(hits).head(rag_topk_docs)
            top_docs      = agg_df["image_uuid"].tolist()
            selected_hits = select_chunks_for_top_docs(hits, top_docs, per_doc=3)
            prompt, source_map = build_prompt_and_source_map(q, selected_hits)

            with st.spinner("Generating answer with Gemini…"):
                answer = gemini_answer(prompt, rag_llm_model.strip())

            st.session_state.rag_answer       = answer
            st.session_state.rag_hits         = selected_hits
            st.session_state.rag_doc_rank     = agg_df
            st.session_state.rag_source_map   = source_map
            st.session_state.rag_selected_doc = top_docs[0] if top_docs else None

            st.session_state.chat_history.append({
                "role":    "assistant",
                "text":    answer,
                "sources": [s["image_uuid"] for s in source_map[:5]],
            })
        st.rerun()

    if st.session_state.rag_source_map:
        st.divider()
        st.markdown("<span class='section-label'>Sources &amp; Document Viewer</span>", unsafe_allow_html=True)
        left, right = st.columns([1, 3], gap="large")

        with left:
            st.markdown("**Document Viewer**")
            if not st.session_state.rag_doc_rank.empty:
                docs = st.session_state.rag_doc_rank["image_uuid"].tolist()
                if st.session_state.rag_selected_doc not in docs:
                    st.session_state.rag_selected_doc = docs[0]
                sel = st.selectbox("Select document", docs, key="rag_doc_view")
                st.session_state.rag_selected_doc = sel
                p = find_image_path(sel, cfg.image_base_dir, extra_dirs=[DEFAULT_UPLOAD_DIR])
                if p: show_image(p, caption=sel, use_container_width=True)
                else: st.warning("Image not found on disk.")

        with right:
            st.markdown("**Retrieved Documents (ranked by distance)**")
            if not st.session_state.rag_doc_rank.empty:
                st.dataframe(
                    st.session_state.rag_doc_rank, use_container_width=True,
                    column_config={
                        "best_distance": st.column_config.NumberColumn(format="%.6f"),
                        "hit_count":     st.column_config.NumberColumn(format="%d"),
                    },
                )

            st.markdown("**Reference Legend** — maps [S1], [S2]… to source documents")
            refs_df = pd.DataFrame(st.session_state.rag_source_map)
            st.dataframe(
                refs_df[["source", "image_uuid", "chunk_id", "cosine_distance", "preview"]],
                use_container_width=True,
                column_config={"cosine_distance": st.column_config.NumberColumn(format="%.6f")},
            )

            if rag_show_sources and st.session_state.rag_hits:
                st.markdown("**Full Chunk Text**")
                for i, (chunk_id, image_uuid, chunk_text, dist) in enumerate(st.session_state.rag_hits, 1):
                    with st.expander(f"[S{i}]  {image_uuid}  ·  {chunk_id}  ·  dist={float(dist):.4f}"):
                        st.write(chunk_text)


# ─────────────────────────────────────────────
# PIPELINE WORKER  (runs in background thread)
# ─────────────────────────────────────────────

def _run_pipeline(
    image_dir: str,
    saved_files: list,
    log_q: queue.Queue,
    custom_embedder,
    layoutlmv3_embedder,
) -> None:
    """
    Background thread: runs all embedding stages using pre-loaded embedder objects.
    Pushes (stage_key, msg, level) log tuples and ("__PROGRESS__", n_done, n_total)
    progress updates onto log_q, then signals ("__DONE__", success).
    """
    def log(stage: str, msg: str, level: str = "info"):
        log_q.put((stage, msg, level))

    def progress(n_done: int, n_total: int):
        log_q.put(("__PROGRESS__", n_done, n_total))

    N = 3  # total stages: custom, layoutlmv3, merge+chunks
    out_dir = os.path.join(os.path.dirname(image_dir.rstrip("/\\")), "embeddings")
    os.makedirs(out_dir, exist_ok=True)
    all_dfs = []

    try:
        import re as _re
        import psycopg2 as _psycopg2
        from utils.db import get_db_params as _get_db_params

        FLOAT_RE = _re.compile(r"-?(?:\d+\.\d+|\d+\.|\.\d+|\d+)(?:[eE]-?\d+)?")

        def _to_pgvec(vec_str: str) -> str:
            """Convert any vector string representation to pgvector literal."""
            if not isinstance(vec_str, str) or not vec_str.strip():
                return None
            nums = FLOAT_RE.findall(vec_str)
            return "[" + ",".join(nums) + "]" if nums else None

        def _upsert_vectors(df: pd.DataFrame, col_map: dict, log_stage: str):
            """
            Upsert rows from df into suki_fyp_vectors_v2.
            col_map: { df_column -> db_column }
            colpali_embeddings is always NULL for uploaded images.
            """
            try:
                conn = _psycopg2.connect(**_get_db_params())
                cur  = conn.cursor()
                cur.execute("SET statement_timeout = '10min';")
                cur.execute("SET lock_timeout = '2s';")

                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {cfg.table_name} (
                        image_uuid           TEXT PRIMARY KEY,
                        colpali_embeddings   VECTOR(768),
                        text_embedding       VECTOR(384),
                        layout_embedding     VECTOR(2500),
                        combined_embedding   VECTOR(2884),
                        layoutlmv3_embedding VECTOR(768),
                        document_level       TEXT,
                        template_level       TEXT
                    );
                """)
                conn.commit()

                inserted = 0
                for _, row in df.iterrows():
                    image_uuid = str(row.get("image_uuid", "")).strip()
                    if not image_uuid:
                        continue

                    sets   = []
                    values = []
                    for df_col, db_col in col_map.items():
                        vec = _to_pgvec(str(row.get(df_col, "") or ""))
                        if vec:
                            sets.append(f"{db_col} = %s::vector")
                            values.append(vec)

                    if not sets:
                        continue

                    values.append(image_uuid)
                    cur.execute(f"""
                        INSERT INTO {cfg.table_name} (image_uuid)
                        VALUES (%s)
                        ON CONFLICT (image_uuid) DO NOTHING;
                    """, (image_uuid,))
                    cur.execute(f"""
                        UPDATE {cfg.table_name}
                        SET {", ".join(sets)}
                        WHERE image_uuid = %s;
                    """, values)
                    inserted += 1

                conn.commit()
                cur.close()
                conn.close()
                log(log_stage, f"✓ {inserted} row(s) upserted into `{cfg.table_name}`", "ok")

            except Exception as e:
                log(log_stage, f"⚠ DB upsert failed: {e}", "warn")

        # ── Stage 1: Custom Embedder ─────────────────────────────────────────
        log("custom", "Running Custom Embedder (SentenceTransformer + layout grid)…")
        try:
            if custom_embedder is None:
                raise RuntimeError("CustomEmbedder failed to load at startup — check configs.py logs.")
            custom_embedder.image_folder = image_dir
            custom_embedder.image_files  = [
                f for f in os.listdir(image_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            df = custom_embedder.generate_embeddings(desc="Custom")
            if not df.empty:
                csv_path = os.path.join(out_dir, "custom_embeddings.csv")
                df.to_csv(csv_path, index=False)
                all_dfs.append(df)
                log("custom", f"✓ {len(df)} row(s) saved → `{csv_path}`", "ok")
                # Insert into vectors table — colpali_embeddings left NULL
                _upsert_vectors(df, {
                    "text_embedding":    "text_embedding",
                    "layout_embedding":  "layout_embedding",
                    "combined_embedding":"combined_embedding",
                }, "custom")
            else:
                log("custom", "⚠ Custom Embedder returned an empty DataFrame.", "warn")
        except Exception as e:
            log("custom", f"✗ Custom Embedder failed: {e}", "error")
        progress(1, N)

        # ── Stage 2: LayoutLMv3 ──────────────────────────────────────────────
        log("layoutlm", "Running LayoutLMv3 Embedder…")
        try:
            if layoutlmv3_embedder is None:
                raise RuntimeError("LayoutLMv3 failed to load at startup — check configs.py logs.")
            layoutlmv3_embedder.image_folder = image_dir
            layoutlmv3_embedder.image_files  = [
                f for f in os.listdir(image_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            df = layoutlmv3_embedder.generate_embeddings(desc="LayoutLMv3")
            if not df.empty:
                csv_path = os.path.join(out_dir, "layoutlmv3_embeddings.csv")
                df.to_csv(csv_path, index=False)
                all_dfs.append(df)
                log("layoutlm", f"✓ {len(df)} row(s) saved → `{csv_path}`", "ok")
                # Insert into vectors table
                _upsert_vectors(df, {
                    "layoutlmv3_embedding": "layoutlmv3_embedding",
                }, "layoutlm")
            else:
                log("layoutlm", "⚠ LayoutLMv3 returned an empty DataFrame.", "warn")
        except Exception as e:
            log("layoutlm", f"✗ LayoutLMv3 Embedder failed: {e}", "error")
        progress(2, N)

        # ── Stage 3: Merge CSVs + Text Chunking placeholder ──────────────────
        log("chunks", "Merging all embedding CSVs…")
        try:
            if all_dfs:
                merged = all_dfs[0]
                for df in all_dfs[1:]:
                    merged = pd.merge(merged, df, on="image_uuid", how="outer")
                combined_path = os.path.join(out_dir, "combined_embeddings.csv")
                merged.to_csv(combined_path, index=False)
                log("chunks", f"✓ Combined CSV saved — {len(merged)} row(s) → `{combined_path}`", "ok")
            else:
                log("chunks", "⚠ No embeddings to merge.", "warn")
        except Exception as e:
            log("chunks", f"✗ Merge failed: {e}", "error")

        # ── Stage 3: OCR → Chunk → Embed → CSV + pgvector insert ────────────
        log("chunks", "Starting OCR + chunking + embedding + DB insert…")
        try:
            import re
            import pytesseract
            import psycopg2
            from psycopg2.extras import execute_values
            from utils.db import get_db_params

            DOC_TABLE    = cfg.table_name
            CHUNK_TABLE  = cfg.chunk_table_name
            EMBED_DIMS   = 384
            CHUNK_CHARS  = 1200
            OVERLAP      = 150
            EMBED_BATCH  = 128
            INSERT_BATCH = 2000

            def _clean_text(t: str) -> str:
                if not isinstance(t, str):
                    return ""
                t = t.replace("\x0c", " ")
                t = re.sub(r"\s+", " ", t)
                return t.strip()

            def _chunk_text(text: str):
                if not text:
                    return []
                chunks, start, n = [], 0, len(text)
                while start < n:
                    end = min(n, start + CHUNK_CHARS)
                    chunks.append((start, end, text[start:end]))
                    if end == n:
                        break
                    start = end - OVERLAP
                return chunks

            def _to_vec_literal(vec):
                return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

            def _embed_batch(texts):
                return embedder.encode(
                    texts,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                ).tolist()

            # ── DB setup ─────────────────────────────────────────────────────
            conn = psycopg2.connect(**get_db_params())
            cur  = conn.cursor()
            cur.execute("SET statement_timeout = '10min';")
            cur.execute("SET lock_timeout = '2s';")
            conn.commit()

            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {CHUNK_TABLE} (
                    chunk_id        TEXT PRIMARY KEY,
                    image_uuid      TEXT NOT NULL,
                    chunk_index     INT  NOT NULL,
                    chunk_text      TEXT NOT NULL,
                    chunk_embedding VECTOR({EMBED_DIMS}),
                    char_start      INT,
                    char_end        INT
                );
            """)
            cur.execute(f"""
                ALTER TABLE {CHUNK_TABLE}
                DROP CONSTRAINT IF EXISTS {CHUNK_TABLE}_image_uuid_fkey;
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {CHUNK_TABLE}_embedding_idx
                ON {CHUNK_TABLE}
                USING ivfflat (chunk_embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            conn.commit()

            all_rows     = []
            db_buffer    = []
            total_chunks = 0

            for fname in saved_files:
                image_uuid = os.path.splitext(fname)[0]
                image_path = os.path.join(image_dir, fname)
                log("chunks", f"OCR → {fname}…")

                try:
                    raw_text = pytesseract.image_to_string(Image.open(image_path))
                    text = _clean_text(raw_text)
                except Exception as e:
                    log("chunks", f"⚠ OCR failed for {fname}: {e}", "warn")
                    text = ""

                if not text:
                    log("chunks", f"⚠ No text extracted from {fname}.", "warn")
                    continue

                # Update extracted_text on vectors table if row exists
                try:
                    cur.execute(f"ALTER TABLE {DOC_TABLE} ADD COLUMN IF NOT EXISTS extracted_text TEXT;")
                    cur.execute(f"UPDATE {DOC_TABLE} SET extracted_text = %s WHERE image_uuid = %s;", (text, image_uuid))
                    conn.commit()
                except Exception:
                    conn.rollback()

                chunks      = _chunk_text(text)
                chunk_texts = [c[2] for c in chunks]
                if not chunk_texts:
                    continue

                log("chunks", f"Embedding {len(chunk_texts)} chunk(s) for {fname}…")
                embs = []
                for i in range(0, len(chunk_texts), EMBED_BATCH):
                    embs.extend(_embed_batch(chunk_texts[i:i + EMBED_BATCH]))

                if len(embs) != len(chunks):
                    log("chunks", f"⚠ Embedding count mismatch for {fname} — skipping.", "warn")
                    continue

                for idx, ((s, e, ct), emb) in enumerate(zip(chunks, embs)):
                    vec_lit = _to_vec_literal(emb)
                    chunk_id = f"{image_uuid}::{idx:04d}"
                    all_rows.append({
                        "chunk_id":        chunk_id,
                        "image_uuid":      image_uuid,
                        "chunk_index":     idx,
                        "chunk_text":      ct,
                        "chunk_embedding": vec_lit,
                        "char_start":      s,
                        "char_end":        e,
                    })
                    db_buffer.append((chunk_id, image_uuid, idx, ct, vec_lit, s, e))
                    total_chunks += 1

                    if len(db_buffer) >= INSERT_BATCH:
                        execute_values(cur, f"""
                            INSERT INTO {CHUNK_TABLE}
                                (chunk_id, image_uuid, chunk_index, chunk_text,
                                 chunk_embedding, char_start, char_end)
                            VALUES %s
                            ON CONFLICT (chunk_id) DO UPDATE
                            SET chunk_text      = EXCLUDED.chunk_text,
                                chunk_embedding = EXCLUDED.chunk_embedding,
                                char_start      = EXCLUDED.char_start,
                                char_end        = EXCLUDED.char_end;
                        """, db_buffer, page_size=1000)
                        db_buffer.clear()
                        conn.commit()

                preview = chunks[0][2][:100].replace("\n", " ")
                log("chunks", f"✓ {fname} → {len(chunks)} chunk(s). Preview: \"{preview}…\"", "ok")

            # Flush remainder
            if db_buffer:
                execute_values(cur, f"""
                    INSERT INTO {CHUNK_TABLE}
                        (chunk_id, image_uuid, chunk_index, chunk_text,
                         chunk_embedding, char_start, char_end)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO UPDATE
                    SET chunk_text      = EXCLUDED.chunk_text,
                        chunk_embedding = EXCLUDED.chunk_embedding,
                        char_start      = EXCLUDED.char_start,
                        char_end        = EXCLUDED.char_end;
                """, db_buffer, page_size=1000)
                conn.commit()

            cur.execute(f"ANALYZE {CHUNK_TABLE};")
            conn.commit()
            cur.close()
            conn.close()

            # Also save to CSV for inspection
            if all_rows:
                csv_path = os.path.join(out_dir, "chunks.csv")
                pd.DataFrame(all_rows).to_csv(csv_path, index=False)
                log("chunks", f"✓ {total_chunks} chunk(s) inserted into `{CHUNK_TABLE}` and saved → `{csv_path}`", "ok")
            else:
                log("chunks", "⚠ No chunks produced.", "warn")

        except Exception as e:
            log("chunks", f"✗ Chunking/insert failed: {e}", "error")

        progress(3, N)
        log_q.put(("__DONE__", True))

    except Exception as e:
        log_q.put(("__ERROR__", str(e)))
        log_q.put(("__DONE__", False))


# ─────────────────────────────────────────────
# PAGE: UPLOAD
# ─────────────────────────────────────────────

def page_upload():
    st.markdown("<h1 style='font-size:22px;font-weight:700;letter-spacing:-0.02em;color:#1A1916;margin-bottom:4px'>Upload Document</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:13px;color:#6B6860;margin-bottom:24px'>Upload document images — the full embedding pipeline runs automatically in the background.</p>", unsafe_allow_html=True)

    # Pipeline stage definitions (label, session key for log lines)
    PIPELINE_STAGES = [
        ("custom",   "1. Custom Embedder",        "text + layout + combined embeddings"),
        ("layoutlm", "2. LayoutLMv3 Embedder",     "multimodal layout + text embeddings"),
        ("chunks",   "3. Text Chunking + pgvector", "chunk text → insert all embeddings to DB"),
    ]

    stage = st.session_state.upload_stage

    # ── IDLE: file uploader ───────────────────────────────────────────────────
    if stage == "idle":
        uploaded_files = st.file_uploader(
            "Drop document images here (.png / .jpg / .jpeg)",
            type=["png", "jpg", "jpeg"], accept_multiple_files=True,
        )
        overwrite = st.checkbox("Overwrite if filename already exists", value=False)

        # Pipeline preview
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.write("**Pipeline — runs automatically after upload:**")
        for key, label, desc in PIPELINE_STAGES:
            st.markdown(f"- **{label}** — {desc}")

        if uploaded_files and st.button("Upload & Run Pipeline →", type="primary"):
            saved, failed = [], []
            for uf in uploaded_files:
                try:
                    fname = save_uploaded_image(uf, DEFAULT_UPLOAD_DIR, overwrite)
                    saved.append(fname)
                except Exception as e:
                    failed.append((uf.name, str(e)))

            for name, err in failed:
                st.error(f"Failed to save: {name} — {err}")

            if saved:
                # Store state needed during processing
                st.session_state.upload_file_name  = ", ".join(saved)
                st.session_state.upload_saved_list = saved
                st.session_state.upload_stage      = "processing"
                st.session_state.pipeline_log      = []
                st.session_state.pipeline_progress = 0
                st.session_state.pipeline_done     = False
                st.session_state.pipeline_success  = False

                # Pre-load embedders now (cached — won't reload if already loaded)
                custom_emb   = load_custom_embedder(DEFAULT_UPLOAD_DIR)
                layoutlm_emb = load_layoutlmv3_embedder(DEFAULT_UPLOAD_DIR)

                # Start background thread with pre-loaded embedders
                log_q: queue.Queue = queue.Queue()
                st.session_state.pipeline_queue = log_q
                t = threading.Thread(
                    target=_run_pipeline,
                    args=(DEFAULT_UPLOAD_DIR, saved, log_q,
                          custom_emb, layoutlm_emb),
                    daemon=True,
                )
                t.start()
                st.rerun()

    # ── PROCESSING: live log ─────────────────────────────────────────────────
    elif stage == "processing":
        fname = st.session_state.get("upload_file_name", "document")

        with st.container(border=True):
            st.markdown(f"**{fname}**")
            st.caption(f"Saved to `{DEFAULT_UPLOAD_DIR}` — pipeline running in the background.")

        # Progress bar — updated by the global banner drain above
        n_stages     = 3
        progress_val = st.session_state.get("pipeline_progress", 0)
        st.progress(progress_val / n_stages, text=f"Stage {progress_val} / {n_stages} complete")

        # Render log lines grouped by stage (populated by the banner drain)
        current_stage_logs: dict = {}
        for (skey, msg, level) in st.session_state.pipeline_log:
            current_stage_logs.setdefault(skey, []).append((msg, level))

        for key, label, desc in PIPELINE_STAGES:
            logs = current_stage_logs.get(key, [])
            if not logs:
                st.markdown(f"⏳ **{label}** — waiting…")
                continue
            last_level = logs[-1][1]
            icon = "✅" if last_level == "ok" else ("❌" if last_level == "error" else "⚙️")
            with st.expander(f"{icon} {label}", expanded=(last_level not in ("ok", "error"))):
                for msg, level in logs:
                    if level == "ok":      st.success(msg)
                    elif level == "error": st.error(msg)
                    elif level == "warn":  st.warning(msg)
                    else:                  st.write(f"`{msg}`")

        st.caption("Navigate to any other page freely — the pipeline keeps running. Come back here or click ↻ Refresh in the banner to check progress.")
        if st.button("↻ Refresh log", key="upload_refresh"):
            st.rerun()

    # ── DONE ────────────────────────────────────────────────────────────────
    elif stage == "done":
        fname   = st.session_state.get("upload_file_name", "document")
        success = st.session_state.get("pipeline_success", False)
        uploaded_uuids = [
            os.path.splitext(f)[0]
            for f in st.session_state.get("upload_saved_list", [])
        ]

        if success:
            st.success(f"✓ Pipeline complete for **{fname}**")
        else:
            st.warning(f"Pipeline finished with errors for **{fname}** — check logs below.")

        # ── Q&A about the uploaded document ──────────────────────────────────
        st.markdown("---")
        st.markdown("### Ask a question about this document")
        st.caption(f"Queries the chunks extracted from: **{fname}**")

        query = st.text_input(
            "Your question",
            value=st.session_state.upload_qa_query,
            placeholder="e.g. What is the main topic of this document?",
            key="upload_qa_input",
        )

        c_ask, c_clear = st.columns([1, 5])
        with c_ask:
            ask = st.button("Ask →", type="primary", key="upload_qa_ask")
        with c_clear:
            if st.button("Clear", key="upload_qa_clear"):
                st.session_state.upload_qa_query  = ""
                st.session_state.upload_qa_answer = None
                st.session_state.upload_qa_hits   = []
                st.rerun()

        if ask and query.strip():
            st.session_state.upload_qa_query = query.strip()
            with st.spinner("Retrieving chunks and generating answer…"):
                try:
                    # Retrieve chunks scoped to the uploaded image_uuids
                    qvec = embed_query_vec(query.strip())
                    qlit = to_vec_literal(qvec)
                    uuid_filter = ""
                    params: tuple
                    if uploaded_uuids:
                        placeholders = ",".join(["%s"] * len(uploaded_uuids))
                        uuid_filter  = f"AND image_uuid IN ({placeholders})"
                        params = (qlit, *uploaded_uuids, qlit, 5)
                    else:
                        params = (qlit, qlit, 5)

                    sql = f"""
                        SELECT chunk_id, image_uuid, chunk_text,
                               (chunk_embedding <=> %s::vector) AS cosine_distance
                        FROM   {cfg.chunk_table_name}
                        WHERE  chunk_embedding IS NOT NULL
                        {uuid_filter}
                        ORDER  BY chunk_embedding <=> %s::vector
                        LIMIT  %s;
                    """
                    hits, _ = run_query(sql, params)
                    hits = hits or []

                    if not hits:
                        st.session_state.upload_qa_answer = "⚠ No chunks found for this document — try re-uploading or check the pipeline log."
                        st.session_state.upload_qa_hits   = []
                    else:
                        prompt, _ = build_prompt_and_source_map(query.strip(), hits)
                        answer     = gemini_answer(prompt, cfg.llm_model)
                        st.session_state.upload_qa_answer = answer
                        st.session_state.upload_qa_hits   = hits
                except Exception as e:
                    st.session_state.upload_qa_answer = f"✗ Error: {e}"
                    st.session_state.upload_qa_hits   = []

        if st.session_state.upload_qa_answer:
            st.markdown("**Answer**")
            st.write(st.session_state.upload_qa_answer)

            if st.session_state.upload_qa_hits:
                with st.expander("View retrieved chunks"):
                    for i, (chunk_id, image_uuid, chunk_text, dist) in enumerate(st.session_state.upload_qa_hits, 1):
                        st.markdown(f"**[S{i}]** `{chunk_id}` — dist `{float(dist):.4f}`")
                        st.write(chunk_text)
                        st.markdown("---")

        # ── Pipeline log + nav ────────────────────────────────────────────────
        st.markdown("---")
        if st.session_state.get("pipeline_log"):
            with st.expander("View full pipeline log"):
                for skey, msg, level in st.session_state.pipeline_log:
                    if level == "ok":      st.success(msg)
                    elif level == "error": st.error(msg)
                    elif level == "warn":  st.warning(msg)
                    else:                  st.write(f"`{msg}`")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            if st.button("Go to Similarity Search →"):
                if uploaded_uuids:
                    st.session_state.sim_anchor_uuid = uploaded_uuids[0]
                    st.session_state.sim_results     = None
                st.session_state.upload_stage = "idle"
                st.session_state.page = "Similarity"
                st.rerun()
        with c2:
            if st.button("Upload More"):
                st.session_state.upload_stage     = "idle"
                st.session_state.upload_qa_query  = ""
                st.session_state.upload_qa_answer = None
                st.session_state.upload_qa_hits   = []
                st.rerun()



# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────

_page = st.session_state.page
if   _page == "Dashboard":  page_dashboard()
elif _page == "Library":    page_library()
elif _page == "Similarity": page_similarity()
elif _page == "RAG":        page_rag()
elif _page == "Upload":     page_upload()