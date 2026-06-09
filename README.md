# ArXiv Semantic Search MCP

A production-grade semantic search engine for arXiv research papers, exposing a **Model Context Protocol (MCP) Server** that any AI Agent can connect to as a tool.

Built with custom mathematical implementations — no black-box wrappers — to demonstrate deep understanding of the underlying algorithms.

---

## What It Does

User (or AI Agent) asks: *"How do transformers handle long sequences?"*

The system:
1. Searches indexed arXiv papers using **Hybrid Search** (semantic + keyword)
2. Feeds top results into an **LLM via RAG pipeline**
3. Returns a synthesized, citation-grounded answer

---

## Architecture
User Query
↓
SentenceTransformer (PyTorch) → Query Embedding
↓
Hybrid Search:
├── Dense Path:   Cosine Similarity on L2-normalized embeddings
└── Lexical Path: Custom TF-IDF with PyTorch Sparse Tensor
↓
Final Score = α × Dense + (1-α) × Lexical
↓
Top-k Papers → LLM (Groq/Llama 3.1) → Synthesized Answer
↓
MCP Server → AI Agent ready

## Custom Implementations (No Black-Box Wrappers)

| Module | Algorithm | Purpose |
|---|---|---|
| `vector_ops.py` | Cosine Similarity, L2 Normalization | Vector space operations |
| `pca.py` | PCA via SVD | Dimensionality reduction |
| `kmeans.py` | K-Means Clustering | Paper topic grouping |
| `hybrid_search.py` | TF-IDF + Dense Hybrid Scoring | Dual-path retrieval |
| `embedder.py` | SentenceTransformer + arXiv API | Paper fetching & embedding |
| `rag_engine.py` | RAG Pipeline | LLM-powered answer synthesis |
| `mcp_server.py` | MCP Server | AI Agent tool integration |

## Tech Stack

- **PyTorch** — custom math implementations, embeddings
- **SentenceTransformers** — all-MiniLM-L6-v2 (384-dim, CPU-friendly)
- **scipy.sparse** — memory-efficient TF-IDF sparse tensor computation
- **Groq API + Llama 3.1** — RAG answer synthesis
- **MCP Protocol** — AI Agent interoperability
- **arXiv API** — real-time paper fetching

## Project Structure
arxiv-semantic-search-mcp/
├── src/
│   ├── vector_ops.py
│   ├── pca.py
│   ├── kmeans.py
│   ├── hybrid_search.py
│   ├── embedder.py
│   ├── rag_engine.py
│   └── mcp_server.py
├── scripts/
│   └── fetch_sample_data.py
├── data/
├── test_pipeline.py
├── .env.example
└── requirements.txt

## Setup
git clone https://github.com/vythwahh/Arxiv-Semantic-Search-MCP.git
cd Arxiv-Semantic-Search-MCP
pip install -r requirements.txt
cp .env.example .env

Fill in your API key in .env:
GROQ_API_KEY=your_key_here

Get your Groq API key at https://console.groq.com

## How to Run

Fetch sample dataset:
python3 scripts/fetch_sample_data.py

Test pipeline:
python3 test_pipeline.py

Start MCP Server:
python3 src/mcp_server.py

## MCP Tools

| Tool | Description |
|---|---|
| `index_arxiv` | Fetch and index papers on a given topic |
| `search_papers` | Hybrid semantic + keyword search |
| `rag_search` | Search + LLM synthesized answer |

## Sample Dataset

733 unique arXiv papers on LLM topics across 5 subtopics: Large Language Models, LLM Reasoning, LLM Fine-tuning, Prompt Engineering, Retrieval Augmented Generation.
