# 🏠 MortgageIQ — RAG-Powered Mortgage Document Chatbot (v2.1)

A fully local Retrieval-Augmented Generation pipeline that ingests multi-document mortgage PDFs, intelligently segments and classifies each document, preserves tabular structure, and answers natural-language questions with grounded, citation-backed responses — all without a single API key.

---

## Demo

[![Watch Demo Video](https://img.shields.io/badge/▶%20Watch%20Demo-Video-red?style=for-the-badge&logo=loom)](https://www.loom.com/share/eb64fbd3c5ee4a88854d101c6a609bf3)

<div align="center">
  <a href="https://www.loom.com/share/eb64fbd3c5ee4a88854d101c6a609bf3">
    <img src="https://cdn.loom.com/sessions/thumbnails/eb64fbd3c5ee4a88854d101c6a609bf3-with-play.gif" alt="MortgageIQ Demo" width="700"/>
  </a>
</div>

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Evaluation & Metrics](#evaluation--metrics)
- [v2.1 Changelog](#v21-changelog)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)

---

## Problem Statement

Mortgage packets are dense multi-document PDFs — loan agreements, lender fee sheets, pay slips, tax forms, insurance policies — often 50+ pages stapled into a single file. Extracting a specific fee, salary figure, or contract clause means manually hunting through pages of legalese and tables where column alignment is the only thing giving numbers meaning.

MortgageIQ solves this by building a RAG pipeline that **understands document boundaries, preserves table structure, and routes queries to the right document type** before generating an answer.

---

## What It Does

1. **Ingests a multi-document mortgage PDF** and automatically detects where one document ends and another begins (boundary detection via Phi-3).
2. **Classifies each logical document** into one of 11 types (Mortgage Contract, Lender Fee Sheet, Pay Slip, Tax Document, etc.).
3. **Extracts text and tables separately** — tables are converted to structured markdown so the LLM can read column headers, not just a jumble of numbers.
4. **Chunks intelligently** — tables are kept intact as single chunks (never split mid-row), while prose is split with sliding-window overlap.
5. **Embeds and indexes** all chunks in a FAISS vector store with persistence.
6. **Routes queries** — Phi-3 predicts which document type a question targets before retrieval, filtering irrelevant chunks.
7. **Generates grounded answers** with explicit hallucination guards and confidence scoring.
8. **Serves everything through a Gradio UI** with upload, chat, analytics, and pipeline reference tabs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER UPLOADS PDF                             │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
                ┌──────────────────────────────┐
                │  PyMuPDF Text Extraction     │
                │  + pytesseract OCR fallback  │
                │  + find_tables() → Markdown  │
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │  Phi-3 Boundary Detection    │
                │  (anchor page + keyword)     │
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │  Phi-3 Document Classification│
                │  (11 document types)          │
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │  Structure-Aware Chunking    │
                │  Tables intact │ Prose split │
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │  BGE-small-en-v1.5 Embeddings│
                │  → FAISS Flat IP Index       │
                └──────────────┬───────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         QUERY TIME                                   │
│                                                                      │
│  User Question → Phi-3 Route (predict doc_type)                      │
│       → FAISS Retrieval (top-k, cosine threshold)                    │
│           → Phi-3 Answer Generation (table-aware prompt)             │
│               → Hallucination Guard (word-overlap check)             │
│                   → Response + Source Citations + Confidence Score   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Walkthrough

### Step 1 — Install Dependencies

The notebook installs PyTorch, Gradio, PyMuPDF, pytesseract, sentence-transformers, FAISS, and llama-cpp-python (CUDA 12.3 wheels). Tesseract OCR is installed at the system level for scanned page fallback.

### Step 2 — Download & Load Phi-3 Mini

Downloads the Phi-3 Mini 4K Instruct GGUF (Q4 quantization, ~2.2 GB) and loads it with all layers offloaded to GPU (`n_gpu_layers=-1`). This single model handles four tasks: boundary detection, classification, query routing, and answer generation.

### Step 3 — Data Structures & Evaluation Logger

Defines `PageInfo`, `LogicalDocument`, `ChunkMetadata`, and `QueryMetrics` dataclasses. Sets up a per-query evaluation logger that tracks latency, chunk relevance scores, grounding ratios, and confidence throughout the session.

### Step 4 — Document Intelligence

**Classification:** Sends a 600-char sample of each document to Phi-3, which maps it to one of 11 labels (Mortgage Contract, Lender Fee Sheet, Pay Slip, etc.). Falls back to "Other" on empty or unrecognized responses.

**Boundary Detection:** Uses an anchor page strategy — each page is compared against the first page of the current document. When Phi-3 signals a new document (or keyword heuristics trigger), a boundary is recorded and a new logical document begins.

### Step 5 — PDF Extraction with Table Preservation

This is the key v2.1 improvement. Standard text extraction flattens tables into meaningless strings of numbers. MortgageIQ instead:

- Calls `page.find_tables()` on every page to detect tabular regions.
- Converts each detected table to **markdown format** with proper column headers and row delimiters.
- Injects a `--- STRUCTURED TABLE DATA ---` marker so the chunker knows to keep it intact.
- Falls back to regex-based fee line parsing for PDFs where `find_tables()` fails (e.g., Calyx forms).

### Step 6 — Structure-Aware Chunking

Splits on the table marker first — table blocks become individual chunks (never split mid-row). Non-table text uses a sliding window (512 tokens, 100-token overlap). Every chunk carries `doc_type`, `doc_id`, page range, and a `has_table` flag.

### Step 7 — Embeddings & FAISS Index

Encodes all chunks with `BAAI/bge-small-en-v1.5` and stores them in a FAISS Flat Inner Product index. Supports:

- **Relevance threshold** (default 0.35 cosine similarity) — chunks below the threshold are dropped before answer generation.
- **FAISS persistence** — save/load the index and chunk metadata to disk so re-embedding on restart is unnecessary.

### Step 8 — Query Routing & Answer Generation

**Routing:** Phi-3 predicts which `doc_type` a query targets (e.g., "What is my gross pay?" → Pay Slip). Retrieval is filtered to that type, with "All" as a fallback for cross-document questions.

**Answer Generation:** The retrieved chunks are passed to Phi-3 with table-aware prompting — explicit instructions to read column headers before interpreting values. The prompt enforces grounded answers: if the context doesn't contain the answer, the model is instructed to say so.

**Hallucination Guard:** A post-generation check calculates what fraction of non-stopword content words in the answer appear in the retrieved context. Low overlap triggers a warning flag on the response.

### Step 9 — Document Store (Orchestrator)

`MortgageDocumentStore` ties the full pipeline together: PDF → page extraction → boundary detection → classification → chunking → embedding → indexing. Handles FAISS persistence and per-query metrics logging.

### Step 10 — Gradio UI

A four-tab Blocks interface with custom CSS (dark theme):

| Tab | What It Does |
|---|---|
| **Upload** | Drag-and-drop PDF upload, processing stats, document structure cards showing detected documents |
| **Chat** | Conversation panel with a retrieval sidebar displaying source chunks and visual confidence bars |
| **Analytics** | Live per-query performance dashboard — latency, relevance scores, grounding ratios |
| **About** | Full pipeline reference table with v2.1 changelog |

---

## Tech Stack

| Category | Tool |
|---|---|
| LLM | Phi-3 Mini 4K Instruct (GGUF Q4, ~2.2 GB) via llama-cpp-python |
| Embeddings | BGE-small-en-v1.5 (SentenceTransformers) |
| Vector Store | FAISS (Flat Inner Product) |
| PDF Extraction | PyMuPDF (fitz) + pytesseract OCR fallback |
| Table Extraction | PyMuPDF `find_tables()` → Markdown |
| UI Framework | Gradio Blocks |
| Compute | Google Colab T4 GPU |
| Language | Python 3 |

---

## Getting Started

### Prerequisites

- Google Colab with T4 GPU runtime (free tier works), or any CUDA-capable machine
- No API keys required — everything runs locally

### Run on Colab

1. Open the notebook in Google Colab.
2. Set the runtime to **GPU → T4**.
3. Run all cells sequentially (Step 1 through Step 10).
4. The final cell launches a Gradio app with a **public share link** — click it to open the UI in a browser tab.

### Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/MortgageIQ.git
cd MortgageIQ

# Install dependencies
pip install torch gradio pymupdf pypdf2 pytesseract Pillow \
    sentence-transformers faiss-cpu numpy pandas \
    llama-index llama-index-embeddings-huggingface

# Install llama-cpp-python with CUDA support
pip install llama-cpp-python==0.2.90 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu123

# Install Tesseract OCR
sudo apt-get install tesseract-ocr

# Run the notebook or convert to script
jupyter notebook MortgageIQ_RAG_v2_1.ipynb
```

---

## Usage

1. **Upload** a mortgage PDF packet (multi-document PDFs work best — the pipeline will detect boundaries automatically).
2. **Check the Upload tab** for processing stats: number of pages, detected documents, classification labels.
3. **Switch to Chat** and ask questions:
   - *"What is my gross monthly income?"*
   - *"What are the total closing costs?"*
   - *"What is the interest rate on my mortgage?"*
   - *"List all lender fees above $500."*
4. **Adjust retrieval settings** in the sidebar — change the document filter, top-k chunks, or relevance threshold.
5. **Check Analytics** for per-query performance metrics.

---

## Evaluation & Metrics

Every query logs the following metrics to the session analytics dashboard:

| Metric | Description |
|---|---|
| **Latency** | End-to-end time from query to response (seconds) |
| **Chunk Relevance** | Cosine similarity scores of retrieved chunks |
| **Grounding Ratio** | Fraction of answer content words found in the retrieved context |
| **Confidence Score** | Query routing confidence from Phi-3 |
| **Hallucination Flag** | Triggered when grounding ratio falls below threshold |

---

## v2.1 Changelog

**Table Structure Preservation (v2.1):**
- `find_tables()` extracts PDF tables as markdown with column headers preserved
- Structure-aware chunking keeps tables intact — never split mid-row
- Table-aware prompting instructs the LLM to read column headers before interpreting values

**Production Guardrails (v2.0):**
- Per-query evaluation metrics with analytics dashboard
- Try/except on every Phi-3 call with graceful degradation
- Hallucination guard via word-overlap grounding check
- Configurable cosine relevance threshold (default 0.35)
- FAISS index persistence (save/load to disk)

---

## Limitations & Future Work

**Current limitations:**
- Phi-3 Mini Q4 has a 4K context window — very long documents may exceed it during answer generation.
- Boundary detection depends on LLM quality — visually similar consecutive documents of the same type can be missed.
- `find_tables()` doesn't handle every PDF table format — Calyx-style forms fall back to regex parsing.
- Single-PDF-at-a-time upload (no batch processing).

**Potential improvements:**
- Swap to a larger context model (Phi-3 Medium or Mistral 7B) for longer answer windows.
- Add multi-PDF upload and cross-document comparison.
- Implement chunk re-ranking (e.g., cross-encoder reranker) for better retrieval precision.
- Add automated evaluation against a labeled Q&A test set.
- Deploy as a persistent Hugging Face Space with GPU.

---

## Project Structure

```
MortgageIQ/
├── MortgageIQ_RAG_v2_1.ipynb   # Full pipeline notebook (run end-to-end)
├── README.md                    # This file
├── assets/
│   └── demo.gif                 # (optional) Demo recording
└── sample_data/
    └── sample_mortgage.pdf      # (optional) Test PDF
```

---

## License

This project is released for educational and portfolio purposes.

---

<p align="center">
  Built with Phi-3 Mini · BGE-small · FAISS · PyMuPDF · Gradio
</p>
