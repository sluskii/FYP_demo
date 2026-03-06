# FYP Demo — Document Similarity Search & RAG Q&A

A Streamlit application for document similarity search and retrieval-augmented generation (RAG) using multimodal vector embeddings stored in PostgreSQL with pgvector.

---

## Overview

This system allows users to upload document images, automatically generate multiple types of vector embeddings, and then explore documents through two core capabilities:

- **Similarity Search** — find visually or semantically similar documents using different embedding types and distance metrics
- **RAG Q&A** — ask natural language questions about documents, answered by Gemini using retrieved text chunks

---

## Architecture

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
          │     → chunk_embedding       │  (384-dim, inserted to chunks table)
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

**Tech stack:**
- **Frontend** — Streamlit
- **Embeddings** — SentenceTransformer (`all-MiniLM-L6-v2`), LayoutLMv3, custom layout grid
- **Vector DB** — PostgreSQL + pgvector
- **OCR** — Tesseract (`pytesseract`)
- **LLM** — Google Gemini (`gemini-2.5-flash`)

---

## Setup

### Prerequisites
- Python 3.12
- PostgreSQL with the `pgvector` extension enabled
- Tesseract installed on your system (`brew install tesseract` on macOS)
- A Google Gemini API key

### Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
cd app
pip install -r requirements.txt
```

### Configuration

Set up your database connection in `vectordb/database.ini`:

```ini
[postgresql]
host=localhost
database=postgres
port=5432
user=your_username
password=your_password
```

Set your Gemini API key as an environment variable:

```bash
export GEMINI_API_KEY=your_api_key_here
```

### Run

```bash
streamlit run test_app.py
```

---

## Features

### Dashboard
Overview of the corpus — total documents, embedding coverage, and quick navigation.

### Library
Browse all indexed documents as an image grid. Click **Find Similar** on any document to jump directly to Similarity Search with that document pre-selected.

### Similarity Search
Select an anchor document and find the most similar documents in the corpus. Configurable options:
- **Embedding type** — Layout-Only, LayoutLMv3, ColPali, Text-Only (Semantic), or Combined (Hybrid)
- **Distance metric** — Cosine, L2, Inner Product, L1, Hamming, Jaccard
- Side-by-side image comparison with distance scores

### Text Q&A (RAG)
Ask natural language questions across the full document corpus. The system retrieves the most relevant text chunks via vector search and generates a cited answer using Gemini.

### Upload
Upload new document images and run the full embedding pipeline automatically in the background — the app remains fully usable while processing. Once complete, query the uploaded document directly from the upload page before navigating elsewhere.