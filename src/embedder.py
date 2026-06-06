import torch
import arxiv
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Paper:
    """Represents a single arXiv paper."""
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    url: str
    published: str


def _clean_text(text: str) -> str:
    """
    Removes newlines and excess whitespace from raw arXiv text.
    Prevents tokenizer confusion from formatting artifacts in API responses.
    """
    return " ".join(text.split())


class ArxivEmbedder:
    """
    Fetches arXiv papers and generates dense vector embeddings
    using a pretrained SentenceTransformer model.

    The embedding encodes both title and abstract to maximize
    semantic coverage per paper.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: SentenceTransformer model to use.
                        'all-MiniLM-L6-v2' is lightweight (80MB) and fast,
                        outputs 384-dim embeddings — suitable for CPU inference.
        """
        # Fix 3: Auto-detect device at init time for seamless CPU/GPU support
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model_name = model_name

    def fetch_papers(self, query: str, max_results: int = 100) -> List[Paper]:
        """
        Fetches papers from arXiv API matching the query.
        Handles network failures gracefully by returning an empty list
        instead of crashing the MCP Server.

        Args:
            query: Search query string
            max_results: Maximum number of papers to fetch

        Returns:
            List of Paper objects, or empty list on network failure.
        """
        print(f"Fetching {max_results} papers for query: '{query}'")

        # Fix 1: Wrap network call in try-except to handle API failures gracefully
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

            papers = []
            for result in client.results(search):
                papers.append(Paper(
                    paper_id=result.entry_id,
                    title=_clean_text(result.title),
                    # Fix 2: Clean abstract to remove \n and excess whitespace
                    abstract=_clean_text(result.summary),
                    authors=[str(a) for a in result.authors],
                    url=result.pdf_url,
                    published=str(result.published.date())
                ))

            print(f"Fetched {len(papers)} papers")
            return papers

        except Exception as e:
            print(f"[ERROR] Failed to fetch papers from arXiv: {e}")
            return []

    def embed_papers(self, papers: List[Paper]) -> Tuple[List[str], torch.Tensor]:
        """
        Generates dense embeddings for a list of papers.
        Encodes title + abstract together for richer semantic representation.
        Embeddings are automatically placed on the correct device (CPU/GPU).

        Returns:
            texts: List of combined title + abstract strings
            embeddings: shape (n_papers, embedding_dim) on self.device
        """
        texts = [
            f"{paper.title}. {paper.abstract}"
            for paper in papers
        ]

        print(f"Generating embeddings for {len(texts)} papers on {self.device}...")

        # Fix 3: device is handled automatically via self.model initialized on correct device
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )

        print(f"Embeddings shape: {embeddings.shape}")
        return texts, embeddings

    def fetch_and_embed(
        self,
        query: str,
        max_results: int = 100
    ) -> Tuple[List[Paper], List[str], torch.Tensor]:
        """
        Convenience method: fetch papers and embed in one call.
        Returns empty structures if fetch fails.

        Returns:
            papers: List of Paper objects (for metadata)
            texts: List of cleaned title + abstract strings (for TF-IDF)
            embeddings: shape (n_papers, embedding_dim)
        """
        papers = self.fetch_papers(query, max_results)

        if not papers:
            print("[WARNING] No papers fetched — returning empty embeddings")
            return [], [], torch.empty(0)

        texts, embeddings = self.embed_papers(papers)
        return papers, texts, embeddings
     

     