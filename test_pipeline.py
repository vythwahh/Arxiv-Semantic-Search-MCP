import sys
sys.path.append("src")

from embedder import ArxivEmbedder
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

n = pipeline.index(query="large language models", max_results=10)
print(f"Indexed: {n} papers")

results = pipeline.retrieve("how do LLMs handle reasoning?")
for r in results:
    print(f"[{r.rank}] {r.paper.title} ({r.score:.4f})")

answer, _ = pipeline.search_and_generate("what are the main challenges in LLM training?")
print(f"\nAnswer:\n{answer}")
