import torch
from typing import List, Tuple
from dataclasses import dataclass
import logging

# Assuming these are imported from your custom modules within the repo
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
    Built with defensive programming and strict hardware-synchronization guardrails.

    Architecture:
        Query -> Embedder -> HybridSearch (Dense + TF-IDF) -> Top-k Context -> LLM -> Answer
    """

    def __init__(
        self,
        embedder: ArxivEmbedder,
        hybrid_search: HybridSearch,
        llm_client,
        model_name: str = "claude-3-5-sonnet-20241022",
        top_k: int = 5,
        max_context_papers: int = 3
    ):
        """
        Initializes the RAG Pipeline.

        Args:
            embedder: An instance of ArxivEmbedder to handle vectorization.
            hybrid_search: An instance of HybridSearch to perform dual-path retrieval.
            llm_client: The initialized LLM API client (e.g., Anthropic Client).
            model_name: The target LLM model string.
            top_k: Number of documents to retrieve via HybridSearch.
            max_context_papers: Number of top documents to feed into the LLM prompt context.
        """
        self.embedder = embedder
        self.hybrid_search = hybrid_search
        self.llm_client = llm_client
        self.model_name = model_name
        self.top_k = top_k
        self.max_context_papers = max_context_papers
        self.papers = []

    def index(self, query: str, max_results: int = 100) -> int:
        """
        Fetches relevant papers from arXiv API and indexes them into the HybridSearch system.

        Returns:
            int: The total number of successfully indexed papers.
        """
        logger.info(f"Indexing papers for source query: '{query}'")

        self.papers, texts, embeddings = self.embedder.fetch_and_embed(
            query=query,
            max_results=max_results
        )

        if not self.papers:
            logger.warning("No papers fetched from arXiv API — index remains empty.")
            return 0

        self.hybrid_search.index(texts, embeddings)
        logger.info(f"Successfully indexed {len(self.papers)} papers into HybridSearch.")
        return len(self.papers)

    def retrieve(self, query: str) -> List[SearchResult]:
        """
        Retrieves top-k relevant papers using the dual-path hybrid search.
        Guards against empty index calls and prevents silent hardware device mismatches.

        Returns:
            List[SearchResult]: A list of wrapped search results sorted by descending scores.
        """
        # Defensive Guardrail: Ensure repository has been indexed prior to querying
        if not self.papers:
            logger.warning("Retrieve called on an empty index. Please run index() first.")
            return []

        # Hardware Sync Optimization: Encode via GPU target device, but instantly 
        # force evaluation back to CPU memory (.cpu()) to perfectly align with 
        # HybridSearch matrices — completely neutralizing PyTorch Device Mismatch crashes.
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
            # Index-bound check to prevent potential out-of-bounds corruption
            if idx < len(self.papers):
                search_results.append(SearchResult(
                    paper=self.papers[idx],
                    score=score,
                    rank=rank + 1
                ))
        return search_results

    def _build_context(self, results: List[SearchResult]) -> str:
        """
        Flattens and formats top retrieved search results into a clean, structured 
        context block for prompt feeding.
        """
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
        Synthesizes an objective, factual response from the LLM based strictly 
        on the retrieved scientific text. Implements anti-hallucination constraints.

        Returns:
            str: Synthesized answer or a clear fallback message if unavailable.
        """
        context = self._build_context(results)

        # Anti-Hallucination Prompting Strategy: Enforcing strict context grounding
        prompt = f"""You are an elite expert scientific research assistant specializing in analyzing arXiv papers.
Your task is to provide a concise, accurate synthesized answer to the user query strictly based on the retrieved context papers provided below.

Query: {query}

Retrieved Papers Context:
{context}

Strict Operational Instructions:
1. Rely ONLY on the clear facts mentioned in the Retrieved Papers Context above. 
2. Do NOT use your own pre-trained knowledge or extrapolate beyond the provided text.
3. If the provided papers do not contain enough information to fully answer the query, state exactly and clearly: "I cannot find sufficient information in the retrieved papers to answer this query."
4. Maintain a formal scientific tone, cite specific paper titles when drawing insights, and keep the final response under 250 words.
"""

        try:
            response = self.llm_client.messages.create(
                model=self.model_name,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        except Exception as e:
            logger.error(f"LLM text generation failed via API: {e}")
            # Graceful software degradation fallback
            if results:
                return f"[System Notice: LLM Generation temporarily unavailable]. Top retrieved source: {results[0].paper.title}"
            return "[System Notice: LLM Generation unavailable and no context documents found.]"

    def search_and_generate(self, query: str) -> Tuple[str, List[SearchResult]]:
        """
        Executes the full end-to-end RAG pipeline process: Retrieval followed by Generation.

        Returns:
            Tuple[str, List[SearchResult]]: Synthesized answer string and the list of source citations used.
        """
        logger.info(f"Executing full end-to-end RAG pipeline execution for query: '{query}'")

        results = self.retrieve(query)

        if not results:
            return "No relevant indexed papers found to process this query. Please ensure the repository is indexed correctly.", []

        answer = self.generate(query, results)
        return answer, results