import sys
import json
sys.path.append("src")

from embedder import ArxivEmbedder, Paper
from hybrid_search import HybridSearch
from rag_engine import RAGPipeline

embedder = ArxivEmbedder()
hybrid_search = HybridSearch(alpha=0.7)
pipeline = RAGPipeline(
    embedder=embedder,
    hybrid_search=hybrid_search,
    top_k=5,
    max_context_papers=3
)

with open("scripts/data/sample_papers.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

papers_to_index = []
for p in raw_data:
    papers_to_index.append(Paper(
        paper_id=p["paper_id"],
        title=p["title"],
        abstract=p["abstract"],
        authors=p["authors"],
        url=p["url"],
        published=p["published"]
    ))

print(f"Total papers loaded from JSON: {len(papers_to_index)}")

texts = [f"{p.title}. {p.abstract}" for p in papers_to_index]
_, embeddings = pipeline.embedder.embed_papers(papers_to_index)

pipeline.papers = papers_to_index
pipeline.hybrid_search.index(texts, embeddings)
print("Data indexing completed successfully.")

print("\n--- RETRIEVAL TEST ---")
results = pipeline.retrieve("how do LLMs handle reasoning?")
for r in results:
    print(f"[{r.rank}] {r.paper.title} ({r.score:.4f})")

print("\n--- RAG GENERATION TEST ---")
answer, _ = pipeline.search_and_generate("what are the main safety alignment techniques for large language models?")
print(f"Answer:\n{answer}")