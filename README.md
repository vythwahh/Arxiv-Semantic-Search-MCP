# ArXiv Semantic Search MCP

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
Custom Implementations (No Black-Box Wrappers)ModuleAlgorithmPurposevector_ops.pyCosine Similarity, L2 NormalizationVector space operationspca.pyPCA via SVDDimensionality reductionkmeans.pyK-Means ClusteringPaper topic groupinghybrid_search.pyTF-IDF + Dense Hybrid ScoringDual-path retrievalembedder.pySentenceTransformer + arXiv APIPaper fetching & embeddingrag_engine.pyRAG Pipeline with Groq/Llama 3.1LLM-powered answer synthesismcp_server.pyMCP ServerAI Agent tool integrationuser_profile.pyEMA + K-Means User ModelingDynamic interest trackingcron_worker.pyAsync Background WorkerProactive paper notificationDynamic User Interest ModelingThe system learns user interests from interactions and proactively surfaces relevant new papers.Interaction weights:Search query: 0.2Paper click: 0.5RAG interaction: 1.0Cold Start (< 10 actions): Uses a single EMA profile vector updated via:$$\mathbf{u}_{\text{new}} = \gamma \cdot \mathbf{u}_{\text{old}} + (1 - \gamma) \cdot \mathbf{v}_{\text{action}}$$Multi-interest (≥ 10 actions): K-Means clusters behavior history into K interest centroids.Background worker: runs hourly, fetches 20 new arXiv papers, computes batch cosine similarity against user interest clusters, and triggers notifications.Tech StackPyTorch – custom math implementations, embeddingsSentenceTransformers – all-MiniLM-L6-v2 (384-dim, CPU-friendly)scipy.sparse` – memory-efficient TF-IDF sparse tensor computationGroq API + Llama 3.1 – RAG answer synthesisMCP Protocol – AI Agent interoperabilityarXiv API – real-time paper fetchingProject StructurePlaintextarxiv-semantic-search-mcp/
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
SetupBashgit clone [https://github.com/vythuwahh/Arxiv-Semantic-Search-MCP.git](https://github.com/vythuwahh/Arxiv-Semantic-Search-MCP.git)
cd Arxiv-Semantic-Search-MCP
pip install -r requirements.txt
cp .env.example .env
Fill in your API key in .env:PlaintextGROQ_API_KEY=your_key_here
Get your Groq API key at console.groq.com.How to RunFetch sample dataset:Bashpython3 scripts/fetch_sample_data.py
Test core pipeline:Bashpython3 test_pipeline.py
Test user behavior modeling:Bashpython3 test_behavior.py
Test background worker:Bashpython3 test_worker.py
Start MCP Server:Bashpython3 src/mcp_server.py
MCP ToolsToolDescriptionindex_arxivFetch and index papers on a given topicsearch_papersHybrid semantic + keyword searchrag_searchSearch + LLM synthesized answerSample Dataset733 unique arXiv papers on LLM topics across 5 subtopics: Large Language Models, LLM Reasoning, Evaluation, Safety, and Vector Databases.