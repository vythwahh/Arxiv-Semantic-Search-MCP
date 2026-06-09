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

### 1. Real-time Search & Inference Pipeline
* **User Query Input**: The system receives a natural language query from the user or AI agent.
* **Vector Embedding Generation**: The query is processed via `SentenceTransformer` (PyTorch) to generate a dense query embedding vector.
* **Dual-Path Hybrid Search Execution**:
  * **Dense Retrieval Path**: Computes exact Cosine Similarity scores against indexed papers using L2-normalized embedding vectors.
  * **Lexical Retrieval Path**: Computes sparse keyword matching scores using a custom TF-IDF implementation powered by PyTorch Sparse Tensors.
* **Hybrid Scoring Fusion**: Integrates both paths to calculate a comprehensive relevance rating based on the formula: `Final Score = α * Dense + (1-α) * Lexical`.
* **RAG Context Synthesis**: Retrieves top-k relevant papers and forwards them as context to the LLM (Groq / Llama 3.1) to generate a synthesized, citation-grounded response.
* **MCP Server Tool Exposure**: Exposes the entire pipeline as a unified protocol ready for direct consumption by any AI Agent system.

### 2. Dynamic User Behavior & Background Notification Layer
* **User Interaction Capture**: Tracks ongoing user behaviors including search queries, document clicks, and RAG interaction logs.
* **Cold-Start Profile Tracking**: Dynamically updates a single user interest profile (`global_profile_tensor`) using an Exponential Moving Average (EMA) formulation.
* **Multi-Interest Topic Clustering**: Triggers K-Means clustering once the history surpasses 10 distinct actions to construct precise topic interest centroids.
* **Asynchronous Cron Worker Processing**: Executes scheduled background scans hourly to fetch newly published arXiv papers, evaluates batch matrix cosine similarities against the user's active interest profiles, and dispatches real-time automated notifications for matching articles.
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

* **arxiv-semantic-search-mcp/**
  * **src/** (Core modules)
    * `vector_ops.py` - Custom cosine similarity and L2 normalization
    * `pca.py` - Custom PCA dimensionality reduction via SVD
    * `kmeans.py` - Custom K-Means clustering algorithm
    * `hybrid_search.py` - Dual-path TF-IDF + Dense hybrid search engine
    * `embedder.py` - arXiv paper fetcher and SentenceTransformer embedding integration
    * `rag_engine.py` - RAG pipeline using Groq and Llama 3.1
    * `mcp_server.py` - Model Context Protocol server exposing core tools
    * `user_profile.py` - Dynamic interest tracking via EMA and K-Means
    * `cron_worker.py` - Async background worker for proactive notifications
  * **scripts/**
    * `fetch_sample_data.py` - Script to download initial dataset
  * **Other Files:**
    * `data/` - Local directory storing indexed papers and vector data
    * `test_pipeline.py` - Integration testing for the search path
    * `test_behavior.py` - Testing user interaction and profile generation
    * `test_worker.py` - Verification script for the background simulation
    * `.env.example` - Template for environmental variables
    * `requirements.txt` - Python package dependencies
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
