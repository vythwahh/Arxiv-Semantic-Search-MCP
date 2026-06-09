 # Arxiv Semantic Search MCP

A production-grade semantic search engine for arXiv research papers, exposing a Model Context Protocol (MCP) Server. Built with custom mathematical implementations — no black-box wrappers — to demonstrate deep understanding of underlying vector space mechanics.

---

## What It Does

User (or AI Agent) asks: *"How do transformers handle long sequences?"*

The system:
1. Searches indexed arXiv papers using **Hybrid Search** (semantic + keyword).
2. Feeds top results into an **LLM via RAG pipeline**.
3. Returns a synthesized, citation-grounded answer.
4. **Learns user interests over time** and proactively notifies when relevant new papers are published.

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
