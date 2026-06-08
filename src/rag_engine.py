import torch
import os
import logging
import google.generativeai as genai
from typing import List, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

from embedder import ArxivEmbedder, Paper
from hybrid_search import HybridSearch

load_dotenv()
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
    Built with defensive programming and strict hardware-synchronization guardrails.

    Architecture:
        Query -> Embedder -> HybridSearch (Dense + TF-IDF) -> Top-k Context -> LLM -> Answer
    """

    def __init__(
        self,
        embedder: ArxivEmbedder,
        hybrid_search: HybridSearch,
        model_name: str = "gemini-1.5-flash",
        top_k: int = 5,
        max_context_papers: int = 3
    ):
        self.embedder = embedder
        self.hybrid_search = hybrid_search
        self.model_name = model_name
        self.top_k = top_k
        self.max_context_papers = max_context_papers
        self.papers = []

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.llm = genai.GenerativeModel(self.model_name)

    def index(self, query: str, max_results: int = 100) -> int:
        """
        Fetches relevant papers from arXiv API and indexes them into HybridSearch.

        Returns:
            int: Total number of successfully indexed papers.
        """
        logger.info(f"Indexing papers for query: '{query}'")

        self.papers, texts, embeddings = self.embedder.fetch_and_embed(
            query=query,
            max_results=max_results
        )

        if not self.papers:
            logger.warning("No papers fetched — index remains empty.")
            return 0

        self.hybrid_search.index(texts, embeddings)
        logger.info(f"Successfully indexed {len(self.papers)} papers.")
        return len(self.papers)

    def retrieve(self, query: str) -> List[SearchResult]:
        """
        Retrieves top-k relevant papers using hybrid search.
        Guards against empty index and device mismatch.

        Returns:
            List[SearchResult]: Search results sorted by descending score.
        """
        if not self.papers:
            logger.warning("Retrieve called on empty index — run index() first.")
            return []

        # Encode on GPU if available, force to CPU to match HybridSearch matrix device
        query_embedding = self.embedder.model.encode(
            query,
            convert_to_tensor=True,
            device=self.embedder.device
        ).cpu()

        results = self.hybrid_search.search(
            query=query,
            query_embedding=query_embedding,
            top_k=self.top_k
        )

        search_results = []
        for rank, (idx, score) in enumerate(results):
            if idx < len(self.papers):
                search_results.append(SearchResult(
                    paper=self.papers[idx],
                    score=score,
                    rank=rank + 1
                ))
        return search_results

    def _build_context(self, results: List[SearchResult]) -> str:
        """Formats top retrieved papers into structured context for LLM prompt."""
        context_parts = []
        for result in results[:self.max_context_papers]:
            context_parts.append(
                f"[Paper {result.rank}] Title: {result.paper.title}\n"
                f"Authors: {', '.join(result.paper.authors[:3])}\n"
                f"Abstract: {result.paper.abstract}\n"
                f"URL: {result.paper.url}"
            )
        return "\n\n---\n\n".join(context_parts)

    def generate(self, query: str, results: List[SearchResult]) -> str:
        """
        Synthesizes a factual response using Gemini strictly based on
        retrieved abstracts. Implements anti-hallucination constraints.

        Returns:
            str: Synthesized answer or fallback message if LLM unavailable.
        """
        context = self._build_context(results)

        prompt = f"""You are an elite expert scientific research assistant specializing in analyzing arXiv papers.
Your task is to provide a concise, accurate synthesized answer to the user query strictly based on the retrieved context papers provided below.

Query: {query}

Retrieved Papers Context:
{context}

Strict Operational Instructions:
1. Rely ONLY on the clear facts mentioned in the Retrieved Papers Context above.
2. Do NOT use your own pre-trained knowledge or extrapolate beyond the provided text.
3. If the provided papers do not contain enough information, state exactly: "I cannot find sufficient information in the retrieved papers to answer this query."
4. Maintain a formal scientific tone, cite specific paper titles when drawing insights, and keep the response under 250 words.
"""

        try:
            response = self.llm.generate_content(prompt)
            return response.text

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return (
                f"[System Notice: LLM Generation temporarily unavailable]. "
                f"Top retrieved source: {results[0].paper.title if results else 'No results found'}"
            )

    def search_and_generate(self, query: str) -> Tuple[str, List[SearchResult]]:
        """
        Executes full end-to-end RAG pipeline: retrieval then generation.

        Returns:
            Tuple[str, List[SearchResult]]: Answer and source citations used.
        """
        logger.info(f"Running full RAG pipeline for query: '{query}'")

        results = self.retrieve(query)

        if not results:
            return "No relevant indexed papers found. Please ensure the repository is indexed correctly.", []

        answer = self.generate(query, results)
        return answer, results