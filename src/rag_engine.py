import torch
from typing import List, Tuple
from dataclasses import dataclass
import logging

from embedder import ArxivEmbedder, Paper
from hybrid_search import HybridSearch

logger = logging.getLogger("RAGPipeline")


@dataclass
class SearchResult:
    """Represents a single search result with metadata and score."""
    paper: Paper
    score: float
    rank: int


class RAGPipeline:
    """
    Retrieval-Augmented Generation (RAG) pipeline for arXiv semantic search.

    Flow:
        1. User submits a natural language query
        2. HybridSearch retrieves top-k relevant paper abstracts
        3. Retrieved context is fed into an LLM to generate a synthesized answer
        4. Returns both the LLM answer and the source papers

    Architecture:
        Query → Embedder → HybridSearch (Dense + TF-IDF) → Top-k Papers → LLM → Answer
    """

    def __init__(
        self,
        embedder: ArxivEmbedder,
        hybrid_search: HybridSearch,
        llm_client,
        model_name: str = "claude-sonnet-4-20250514",
        top_k: int = 5,
        max_context_papers: int = 3
    ):
        self.embedder = embedder
        self.hybrid_search = hybrid_search
        self.llm_client = llm_client
        self.model_name = model_name
        self.top_k = top_k
        self.max_context_papers = max_context_papers
        self.papers = []

    def index(self, query: str, max_results: int = 100) -> int:
        """
        Fetches papers from arXiv and indexes them for search.

        Returns:
            Number of papers indexed.
        """
        logger.info(f"Indexing papers for query: '{query}'")

        self.papers, texts, embeddings = self.embedder.fetch_and_embed(
            query=query,
            max_results=max_results
        )

        if not self.papers:
            logger.warning("No papers fetched — index is empty")
            return 0

        self.hybrid_search.index(texts, embeddings)
        logger.info(f"Indexed {len(self.papers)} papers")
        return len(self.papers)

    def retrieve(self, query: str) -> List[SearchResult]:
        """
        Retrieves top-k papers using hybrid search.

        Returns:
            List of SearchResult sorted by descending score.
        """
        # Embed the query using the same model
        query_embedding = self.embedder.model.encode(
            query,
            convert_to_tensor=True,
            device=self.embedder.device
        )

        results = self.hybrid_search.search(
            query=query,
            query_embedding=query_embedding,
            top_k=self.top_k
        )

        return [
            SearchResult(
                paper=self.papers[idx],
                score=score,
                rank=rank + 1
            )
            for rank, (idx, score) in enumerate(results)
        ]

    def _build_context(self, results: List[SearchResult]) -> str:
        """
        Builds a structured context string from top retrieved papers
        to feed into the LLM prompt.
        """
        context_parts = []
        for result in results[:self.max_context_papers]:
            context_parts.append(
                f"[Paper {result.rank}] {result.paper.title}\n"
                f"Authors: {', '.join(result.paper.authors[:3])}\n"
                f"Abstract: {result.paper.abstract}\n"
                f"URL: {result.paper.url}"
            )
        return "\n\n---\n\n".join(context_parts)

    def generate(self, query: str, results: List[SearchResult]) -> str:
        """
        Generates a synthesized answer using the LLM with retrieved papers as context.
        Falls back gracefully if LLM call fails.
        """
        context = self._build_context(results)

        prompt = f"""You are a scientific research assistant specializing in arXiv papers.
Based on the following research papers, provide a concise and accurate answer to the query.

Query: {query}

Retrieved Papers:
{context}

Instructions:
- Synthesize insights from the papers above
- Be specific and cite paper titles when relevant
- If the papers do not fully answer the query, say so clearly
- Keep the answer under 300 words
"""

        try:
            response = self.llm_client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"[LLM unavailable] Top result: {results[0].paper.title if results else 'No results found'}"

    def search_and_generate(self, query: str) -> Tuple[str, List[SearchResult]]:
        """
        Full RAG pipeline: retrieve relevant papers and generate a synthesized answer.

        Returns:
            answer: LLM-generated answer string
            results: List of SearchResult used as context
        """
        logger.info(f"Running RAG pipeline for query: '{query}'")

        results = self.retrieve(query)

        if not results:
            return "No relevant papers found for this query.", []

        answer = self.generate(query, results)
        return answer, results