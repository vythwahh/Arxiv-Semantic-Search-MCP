import torch
import os
import time
import logging
from groq import Groq
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
    Built with defensive programming and hardware-synchronization guardrails.
    """

    def __init__(
        self,
        embedder: ArxivEmbedder,
        hybrid_search: HybridSearch,
        model_name: str = "llama-3.1-8b-instant",
        top_k: int = 5,
        max_context_papers: int = 2
    ):
        self.embedder = embedder
        self.hybrid_search = hybrid_search
        self.model_name = model_name
        self.top_k = top_k
        self.max_context_papers = max_context_papers
        self.papers = []
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def index(self, query: str, max_results: int = 100) -> int:
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
        if not self.papers:
            logger.warning("Retrieve called on empty index — run index() first.")
            return []

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
        """Formats top retrieved papers into structured context with strict token trimming."""
        context_parts = []
        for result in results[:self.max_context_papers]:
            trimmed_abstract = result.paper.abstract
            if len(trimmed_abstract) > 500:
                trimmed_abstract = trimmed_abstract[:500] + "..."

            context_parts.append(
                f"[Paper {result.rank}] Title: {result.paper.title}\n"
                f"Authors: {', '.join(result.paper.authors[:2])}\n"
                f"Abstract: {trimmed_abstract}\n"
                f"URL: {result.paper.url}"
            )
        return "\n\n---\n\n".join(context_parts)

    def generate(self, query: str, results: List[SearchResult]) -> str:
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
            time.sleep(1)  
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.2,
            )
            return chat_completion.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            if results:
                return f"[Groq Fallback] Top retrieved source: {results[0].paper.title} (URL: {results[0].paper.url})"
            return "[System Notice: LLM Generation temporarily unavailable]."

    def search_and_generate(self, query: str) -> Tuple[str, List[SearchResult]]:
        logger.info(f"Running full RAG pipeline for query: '{query}'")
        results = self.retrieve(query)
        if not results:
            return "No relevant indexed papers found. Please ensure the repository is indexed correctly.", []
        answer = self.generate(query, results)
        return answer, results