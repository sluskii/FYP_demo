# 🔍📄 Document Similarity Search & RAG Q&A

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://document-similarity-demo-system.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![Gemini](https://img.shields.io/badge/Gemini-2.5--flash-4285F4.svg)](https://ai.google.dev)

[🚀 Live Demo](https://document-similarity-demo-system.streamlit.app/)

</div>

---

## ✨ Features

🔍 **Document Similarity Search**
- **5 Embedding Types** — Layout-Only, LayoutLMv3, ColPali, Text-Only (Semantic), Combined (Hybrid)
- **6 Distance Metrics** — Cosine, L2, Inner Product, L1, Hamming, Jaccard
- **Side-by-side Comparison** — Visual results with distance scores

💬 **Retrieval-Augmented Generation (RAG)**
- **Chunk Retrieval** — pgvector cosine search over extracted text chunks
- **Gemini Answers** — Cited responses grounded in document content
- **Source Transparency** — View retrieved chunks and distances

⬆️ **Document Upload Pipeline**
- **Background Processing** — App stays responsive while pipeline runs
- **Multi-model Embedding** — Custom Embedder + LayoutLMv3 run automatically
- **OCR → Chunk → Embed** — Tesseract extracts text, MiniLM embeds chunks
- **Instant Q&A** — Query your uploaded document immediately after processing

📚 **Document Library**
- **Image Grid** — Browse all indexed documents visually
- **One-click Search** — Jump to Similarity Search from any document

**Technology Stack:**
- **Frontend** — Streamlit
- **Vector DB** — PostgreSQL + pgvector
- **Embeddings** — SentenceTransformer (`all-MiniLM-L6-v2`), LayoutLMv3, custom layout grid
- **OCR** — Tesseract (`pytesseract`)
- **LLM** — Google Gemini (`gemini-2.5-flash`)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit Frontend                 │
│  Dashboard · Library · Similarity · RAG · Upload    │
└────────────────────────┬────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │       Embedding Pipeline     │  (background thread)
          │                             │
          │  1. Custom Embedder         │  SentenceTransformer + layout grid
          │     → text_embedding        │  (384-dim)
          │     → layout_embedding      │  (2500-dim)
          │     → combined_embedding    │  (2884-dim)
          │                             │
          │  2. LayoutLMv3 Embedder     │  Multimodal layout + text
          │     → layoutlmv3_embedding  │  (768-dim)
          │                             │
          │  3. OCR → Chunk → Embed     │  Tesseract + MiniLM-L6-v2
          │     → chunk_embedding       │  (384-dim)
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │        PostgreSQL            │
          │  + pgvector extension        │
          │                             │
          │  suki_fyp_vectors_v2        │  per-image embeddings
          │  suki_fyp_chunks_v1         │  text chunks + embeddings
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │         Gemini API           │  RAG answer generation
          │    gemini-2.5-flash          │
          └─────────────────────────────┘
```

---

## 🚀 Quick Start

### 🔧 Local Development

```bash
# 1️⃣ Clone the repository
git clone https://github.com/YOUR_USERNAME/FYP_demo.git
cd FYP_demo

# 2️⃣ Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3️⃣ Install dependencies
cd app
pip install -r requirements.txt

# 4️⃣ Run the application
streamlit run test_app.py
```

🌐 **Open your browser** to `http://localhost:8501`

> Requires a PostgreSQL instance with pgvector enabled and credentials configured in `vectordb/database.ini`.