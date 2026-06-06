import torch
import math
from collections import Counter
from typing import List, Tuple


class TFIDF:
    """
    TF-IDF (Term Frequency - Inverse Document Frequency) implemented from scratch.

    Mathematical foundation:
        TF(t, d)     = count(t in d) / len(d)
        IDF(t)       = log(N / (1 + df(t)))    # Laplace smoothing to avoid zero division
        TF-IDF(t, d) = TF(t, d) * IDF(t)
    """

    def __init__(self):
        self.idf = {}
        self.vocab = {}
        self.n_docs = 0

    def fit(self, corpus: List[str]) -> "TFIDF":
        self.n_docs = len(corpus)
        df = Counter()

        for doc in corpus:
            tokens = set(doc.lower().split())
            for token in tokens:
                df[token] += 1

        self.idf = {
            term: math.log(self.n_docs / (1 + freq))
            for term, freq in df.items()
        }
        self.vocab = {term: idx for idx, term in enumerate(self.idf.keys())}
        return self

    def transform(self, texts: List[str]) -> torch.Tensor:
        # Dense matrix kept for readability — can be optimized with scipy.sparse for large corpus
        matrix = torch.zeros(len(texts), len(self.vocab), dtype=torch.float32)
        for i, text in enumerate(texts):
            tokens = text.lower().split()
            if not tokens:
                continue
            tf = Counter(tokens)
            for term, count in tf.items():
                if term in self.vocab:
                    j = self.vocab[term]
                    matrix[i, j] = (count / len(tokens)) * self.idf[term]
        return matrix

    def transform_query(self, query: str) -> torch.Tensor:
        return self.transform([query])[0]


class HybridSearch:
    """
    Hybrid Search combining Dense (semantic) and Lexical (TF-IDF) retrieval.

    Final Score = α * Dense_Score + (1 - α) * Lexical_Score

    Both score types use Cosine Similarity on L2-normalized vectors,
    ensuring scores lie in [-1, 1] without requiring additional normalization.
    This prevents one score type from dominating due to scale differences.
    """

    def __init__(self, alpha: float = 0.7):
        """
        Args:
            alpha: Weight for dense retrieval (0.0 = pure TF-IDF, 1.0 = pure dense)
        """
        assert 0.0 <= alpha <= 1.0, "Alpha must be between 0 and 1"
        self.alpha = alpha
        self.tfidf = TFIDF()
        self.tfidf_matrix = None  # L2-normalized at index time
        self.dense_matrix = None  # L2-normalized at index time
        self.corpus_texts = None

    def index(self, texts: List[str], dense_embeddings: torch.Tensor) -> "HybridSearch":
        """
        Indexes corpus with both TF-IDF and dense embeddings.
        Both matrices are L2-normalized at index time to enable efficient
        cosine similarity via dot product during search.

        Args:
            texts: List of document strings
            dense_embeddings: shape (n_docs, embedding_dim)
        """
        self.corpus_texts = texts
        self.tfidf.fit(texts)

        # Build and L2-normalize TF-IDF matrix at index time
        # This ensures lexical scores lie in [0, 1] without per-query normalization
        raw_tfidf = self.tfidf.transform(texts)
        tfidf_norms = torch.norm(raw_tfidf, p=2, dim=1, keepdim=True)
        self.tfidf_matrix = raw_tfidf / tfidf_norms.clamp(min=1e-8)

        # Force 2D shape and L2-normalize dense embeddings
        dense_embeddings = dense_embeddings.view(len(texts), -1).float()
        dense_norms = torch.norm(dense_embeddings, p=2, dim=1, keepdim=True)
        self.dense_matrix = dense_embeddings / dense_norms.clamp(min=1e-8)

        return self

    def search(self, query: str, query_embedding: torch.Tensor, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Searches corpus using hybrid scoring.

        Fix: query_embedding is flattened to 1D to handle both (embed_dim,)
        and (1, embed_dim) inputs from SentenceTransformer — preventing
        silent shape mismatch bugs in matrix multiplication.

        Returns:
            List of (doc_index, score) sorted by descending score.
        """
        # Flatten to 1D to handle (1, embed_dim) or (embed_dim,) inputs
        query_emb_1d = query_embedding.flatten().float()

        # Dense score: cosine similarity via dot product on L2-normalized vectors
        query_dense_norm = query_emb_1d / torch.norm(query_emb_1d, p=2).clamp(min=1e-8)
        dense_scores = self.dense_matrix @ query_dense_norm  # shape: (n_docs,)

        # Lexical score: cosine similarity via TF-IDF dot product
        query_tfidf_raw = self.tfidf.transform_query(query)
        query_tfidf_norm = query_tfidf_raw / torch.norm(query_tfidf_raw, p=2).clamp(min=1e-8)
        lexical_scores = self.tfidf_matrix @ query_tfidf_norm  # shape: (n_docs,)

        # Both scores are in [-1, 1] — safe to combine without additional normalization
        final_scores = self.alpha * dense_scores + (1 - self.alpha) * lexical_scores

        top_indices = torch.argsort(final_scores, descending=True)[:top_k]
        return [(idx.item(), final_scores[idx].item()) for idx in top_indices]