import torch
import math
from collections import Counter
from typing import List, Tuple


class TFIDF:
    """
    TF-IDF (Term Frequency - Inverse Document Frequency) implemented from scratch.
    Optimized with scipy.sparse to prevent Out-Of-Memory (OOM) on large arXiv corpora.

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
        """
        Builds a true PyTorch Sparse COO Tensor directly from scipy.sparse.
        Eliminates .toarray() entirely to avoid OOM on large corpora.
        """
        from scipy.sparse import lil_matrix
        import numpy as np

        # Step 1: Build sparse matrix in LIL format for fast element-wise assignment
        matrix = lil_matrix((len(texts), len(self.vocab)), dtype=np.float32)

        for i, text in enumerate(texts):
            tokens = text.lower().split()
            if not tokens:
                continue
            tf = Counter(tokens)
            for term, count in tf.items():
                if term in self.vocab:
                    j = self.vocab[term]
                    matrix[i, j] = (count / len(tokens)) * self.idf[term]

        # Step 2: Convert to CSR then COO to extract sparse coordinates
        coo = matrix.tocsr().tocoo()

        # Step 3: Build PyTorch Sparse COO Tensor directly from coordinates
        # No .toarray() call — avoids dense RAM allocation entirely
        indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
        values = torch.FloatTensor(coo.data)
        shape = torch.Size(coo.shape)

        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

    def transform_query(self, query: str) -> torch.Tensor:
        """
        Transforms a single query string to a dense TF-IDF vector.
        Uses dense path since a single vector does not benefit from sparsity.

        Only vocab-present tokens are counted to prevent TF score dilution
        from out-of-vocabulary terms (e.g., stopwords, rare tokens).
        """
        tokens = query.lower().split()
        valid_tokens = [t for t in tokens if t in self.vocab]

        if not valid_tokens:
            return torch.zeros(len(self.vocab), dtype=torch.float32)

        tf = Counter(valid_tokens)
        vector = torch.zeros(len(self.vocab), dtype=torch.float32)
        for term, count in tf.items():
            j = self.vocab[term]
            vector[j] = (count / len(valid_tokens)) * self.idf[term]

        return vector


class HybridSearch:
    """
    Hybrid Search combining Dense (semantic) and Lexical (TF-IDF) retrieval.

    Final Score = α * Dense_Score + (1 - α) * Lexical_Score

    Both score types use Cosine Similarity on L2-normalized vectors,
    ensuring scores lie in [-1, 1] without requiring additional normalization.
    This prevents one retrieval type from dominating due to scale differences.
    """

    def __init__(self, alpha: float = 0.7):
        assert 0.0 <= alpha <= 1.0, "Alpha must be between 0 and 1"
        self.alpha = alpha
        self.tfidf = TFIDF()
        self.tfidf_matrix = None  # L2-normalized PyTorch Sparse Tensor
        self.dense_matrix = None  # L2-normalized PyTorch Dense Tensor
        self.corpus_texts = None

    def index(self, texts: List[str], dense_embeddings: torch.Tensor) -> "HybridSearch":
        """
        Indexes corpus with both TF-IDF and dense embeddings.
        L2 normalization is performed directly on sparse values without .to_dense(),
        making this truly memory-safe for arbitrarily large corpora.
        """
        self.corpus_texts = texts
        self.tfidf.fit(texts)

        # Step 1: Build TF-IDF Sparse Tensor
        raw_tfidf = self.tfidf.transform(texts).coalesce()
        indices = raw_tfidf.indices()
        values = raw_tfidf.values()
        n_docs = len(texts)

        # Step 2: Compute per-row L2 norm directly on sparse values
        # Accumulate squared values into their corresponding rows
        row_sums = torch.zeros(n_docs, dtype=torch.float32).index_add_(
            dim=0,
            index=indices[0],
            source=values ** 2
        )
        tfidf_norms = torch.sqrt(row_sums).clamp(min=1e-8)

        # Step 3: Normalize each sparse value by its row's L2 norm
        # No .to_dense() call — fully memory-safe for large corpora
        normalized_values = values / tfidf_norms[indices[0]]
        self.tfidf_matrix = torch.sparse_coo_tensor(
            indices, normalized_values, raw_tfidf.shape, dtype=torch.float32
        ).coalesce()

        # Step 4: L2-normalize dense embeddings
        dense_embeddings = dense_embeddings.view(n_docs, -1).float()
        dense_norms = torch.norm(dense_embeddings, p=2, dim=1, keepdim=True)
        self.dense_matrix = dense_embeddings / dense_norms.clamp(min=1e-8)

        return self

    def search(self, query: str, query_embedding: torch.Tensor, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Searches corpus using hybrid scoring.

        Args:
            query: Raw query string for lexical scoring
            query_embedding: shape (embed_dim,) or (1, embed_dim) — flattened internally
            top_k: Number of results to return

        Returns:
            List of (doc_index, score) sorted by descending hybrid score.
        """
        # Flatten to 1D to handle both (embed_dim,) and (1, embed_dim) inputs
        query_emb_1d = query_embedding.flatten().float()

        # Dense score: cosine similarity via dot product on L2-normalized vectors
        query_dense_norm = query_emb_1d / torch.norm(query_emb_1d, p=2).clamp(min=1e-8)
        dense_scores = self.dense_matrix @ query_dense_norm  # shape: (n_docs,)

        # Lexical score: sparse-dense matrix multiplication for efficiency
        query_tfidf_raw = self.tfidf.transform_query(query)
        query_tfidf_norm = query_tfidf_raw / torch.norm(query_tfidf_raw, p=2).clamp(min=1e-8)
        lexical_scores = torch.sparse.mm(
            self.tfidf_matrix,
            query_tfidf_norm.view(-1, 1)
        ).flatten()  # shape: (n_docs,)

        # Both scores are in [-1, 1] — safe to combine directly
        final_scores = self.alpha * dense_scores + (1 - self.alpha) * lexical_scores

        top_indices = torch.argsort(final_scores, descending=True)[:top_k]
        return [(idx.item(), final_scores[idx].item()) for idx in top_indices]