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

```
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
Top-k Papers → LLM (Gemini) → Synthesized Answer
    ↓
MCP Server → AI Agent ready
```

---

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

---

## Tech Stack

- **PyTorch** — custom math implementations, embeddings
- **SentenceTransformers** — `all-MiniLM-L6-v2` (384-dim, CPU-friendly)
- **FAISS** — approximate nearest neighbor search
- **scipy.sparse** — memory-efficient TF-IDF computation
- **Gemini API** — RAG answer synthesis
- **MCP Protocol** — AI Agent interoperability
- **arXiv API** — real-time paper fetching

---

## Project Structure

```
arxiv-semantic-search-mcp/
├── src/
│   ├── vector_ops.py       # Custom cosine similarity, L2 normalization
│   ├── pca.py              # Custom PCA via SVD
│   ├── kmeans.py           # Custom K-Means clustering
│   ├── hybrid_search.py    # TF-IDF + dense hybrid search engine
│   ├── embedder.py         # arXiv paper fetcher + SentenceTransformer
│   ├── rag_engine.py       # RAG pipeline with Gemini
│   └── mcp_server.py       # MCP Server exposing 3 tools
├── scripts/
│   └── fetch_sample_data.py  # Fetch sample LLM papers from arXiv
├── data/                     # Auto-generated, gitignored
├── test_pipeline.py          # End-to-end pipeline test
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/vythwahh/Arxiv-Semantic-Search-MCP.git
cd Arxiv-Semantic-Search-MCP
pip install -r requirements.txt
```

Create `.env` in root:
```
GEMINI_API_KEY=your_key_here
```

Get your Gemini API key at [aistudio.google.com](https://aistudio.google.com).

> **Note:** Free tier has rate limits. For production use, upgrade to a paid plan or swap to any OpenAI-compatible API (Groq, OpenAI, Anthropic) by modifying `model_name` in `RAGPipeline`.

---

## How to Run

**Fetch sample dataset:**
```bash
python3 scripts/fetch_sample_data.py
```

**Test pipeline:**
```bash
python3 test_pipeline.py
```

**Start MCP Server:**
```bash
python3 src/mcp_server.py
```

---

## MCP Tools

Once the server is running, any MCP-compatible AI Agent can call:

| Tool | Description |
|---|---|
| `index_arxiv` | Fetch and index papers on a given topic |
| `search_papers` | Hybrid semantic + keyword search |
| `rag_search` | Search + LLM synthesized answer |

---

## Sample Dataset

98 arXiv papers on LLM topics pre-fetched via `scripts/fetch_sample_data.py`:

- Large Language Models
- LLM Reasoning
- LLM Fine-tuning
- Prompt Engineering
- Retrieval Augmented Generation
