# ArXiv Semantic Search MCP

A production-grade semantic search engine for arXiv research papers, exposing a Model Context Protocol (MCP) Server. Built with custom mathematical implementations — no black-box wrappers — to demonstrate deep understanding of underlying vector space mechanics.

---

## What It Does

User (or AI Agent) asks: *"How do transformers handle long sequences?"*

The system:
1. Searches indexed arXiv papers using Hybrid Search (semantic + keyword).
2. Feeds top results into an LLM via RAG pipeline.
3. Returns a synthesized, citation-grounded answer.
4. Learns user interests over time and proactively notifies when relevant new papers are published.

---

## Architecture

```text
User Query
    ↓
SentenceTransformer (PyTorch) → Query Embedding
    ↓
Hybrid Search:
    ├── Dense Path:  Cosine Similarity on L2-normalized embeddings
    └── Lexical Path: Custom TF-IDF with PyTorch Sparse Tensor
    ↓
Final Score = α × Dense + (1-α) × Lexical
    ↓
Top-k Papers → LLM (Groq/Llama 3.1) → Synthesized Answer
    ↓
MCP Server → AI Agent ready

User Behavior Layer (Background):
    User Actions (Search/Click/RAG)
        ↓
    EMA Profile Update → global_profile_tensor
        ↓
    K-Means Clustering (≥10 actions) → interest_centroids
        ↓
    Cron Worker (hourly) → arXiv scan → Batch Cosine Similarity → Notification

---

## Custom Implementations (No Black-Box Wrappers)

* vector_ops.py: Cosine Similarity, L2 Normalization — Vector space operations
* pca.py: PCA via SVD — Dimensionality reduction
* kmeans.py: K-Means Clustering — Paper topic grouping
* hybrid_search.py: TF-IDF + Dense Hybrid Scoring — Dual-path retrieval
* embedder.py: SentenceTransformer + arXiv API — Paper fetching & embedding
* rag_engine.py: RAG Pipeline with Groq/Llama 3.1 — LLM-powered answer synthesis
* mcp_server.py: MCP Server — AI Agent tool integration
* user_profile.py: EMA + K-Means User Modeling — Dynamic interest tracking
* cron_worker.py: Async Background Worker — Proactive paper notification

---

## Dynamic User Interest Modeling

The system learns user interests from interactions and proactively surfaces relevant new papers.

Interaction weights:
* Search query: 0.2
* Paper click: 0.5
* RAG interaction: 1.0

Cold Start (< 10 actions): Uses a single EMA profile vector updated via traditional Exponential Moving Average equations.

Multi-interest (≥ 10 actions): K-Means clusters behavior history into K interest centroids.

Background worker: runs hourly, fetches 20 new arXiv papers, computes batch cosine similarity against user interest clusters, and triggers notifications.

---

## Tech Stack

* PyTorch – custom math implementations, embeddings
* SentenceTransformers – all-MiniLM-L6-v2 (384-dim, CPU-friendly)
* scipy.sparse – memory-efficient TF-IDF sparse tensor computation
* Groq API + Llama 3.1 – RAG answer synthesis
* MCP Protocol – AI Agent interoperability
* arXiv API – real-time paper fetching

---

## Project Structure

arxiv-semantic-search-mcp/
├── src/
│   ├── vector_ops.py       # Custom cosine similarity, L2 normalization
│   ├── pca.py              # Custom PCA via SVD
│   ├── kmeans.py           # Custom K-Means clustering
│   ├── hybrid_search.py    # TF-IDF + dense hybrid search engine
│   ├── embedder.py         # arXiv paper fetcher + SentenceTransformer
│   ├── rag_engine.py       # RAG pipeline with Groq/Llama 3.1
│   ├── mcp_server.py       # MCP Server exposing 3 tools
│   ├── user_profile.py     # EMA user profile + K-Means interest clustering
│   └── cron_worker.py      # Async background worker for paper notifications
├── scripts/
│   └── fetch_sample_data.py
├── data/
├── test_pipeline.py
├── test_behavior.py
├── test_worker.py
├── .env.example
└── requirements.txt

---

## Setup

git clone https://github.com/vythuwahh/Arxiv-Semantic-Search-MCP.git
cd Arxiv-Semantic-Search-MCP
pip install -r requirements.txt
cp .env.example .env

Fill in your API key in .env:
GROQ_API_KEY=your_key_here

Get your Groq API key at console.groq.com.

---

## How to Run

Fetch sample dataset:
python3 scripts/fetch_sample_data.py

Test core pipeline:
python3 test_pipeline.py

Test user behavior modeling:
python3 test_behavior.py

Test background worker:
python3 test_worker.py

Start MCP Server:
python3 src/mcp_server.py

---

## MCP Tools

* index_arxiv: Fetch and index papers on a given topic
* search_papers: Hybrid semantic + keyword search
* rag_search: Search + LLM synthesized answer

---

## Sample Dataset

733 unique arXiv papers on LLM topics across 5 subtopics: Large Language Models, LLM Reasoning, Evaluation, Safety, and Vector Databases.
