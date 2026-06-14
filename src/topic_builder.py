import math
import torch
import sqlite3
import re
from pathlib import Path
from collections import defaultdict
from typing import Optional
from vector_ops import l2_normalize


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "behavior_events.db"

STOPWORDS = {
    'the', 'and', 'for', 'are', 'with', 'that', 'this', 'from', 'how',
    'what', 'when', 'where', 'which', 'has', 'have', 'been', 'was', 'were',
    'their', 'they', 'use', 'using', 'used', 'show', 'shows', 'based',
    'paper', 'propose', 'proposed', 'approach', 'method', 'model', 'models',
    'result', 'results', 'study', 'studies', 'also', 'can', 'its', 'via',
    'our', 'we', 'new', 'two', 'one', 'each', 'such', 'than', 'more',
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class TopicVocab:
    """
    Builds a TF-IDF weighted topic vocabulary from user behavior signals.

    Topics are derived from:
      - keywords logged during search sessions (keyword_events table)
      - abstracts of papers the user clicked and read (paper_interactions table)

    Each topic is represented as a 384-dim vector compatible with
    ArxivEmbedder (all-MiniLM-L6-v2) for downstream cosine similarity.
    """

    def __init__(self, db_path: Path = DB_PATH, top_k: int = 50):
        self.db_path = db_path
        self.top_k = top_k
        self.vocab: list[str] = []
        self.tfidf_scores: dict[str, float] = {}
        self.topic_vectors: Optional[torch.Tensor] = None

    def build(self, user_id: str, embedder=None) -> "TopicVocab":
        """
        Builds topic vocab for a given user.

        Args:
            user_id: Target user.
            embedder: ArxivEmbedder instance. If None, topic_vectors won't be computed.
        """
        keyword_scores = self._load_keyword_scores(user_id)
        abstract_scores = self._load_abstract_scores(user_id)

        merged: dict[str, float] = defaultdict(float)
        for term, score in keyword_scores.items():
            merged[term] += score * 2.0
        for term, score in abstract_scores.items():
            merged[term] += score * 1.0

        sorted_terms = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        self.vocab = [term for term, _ in sorted_terms[: self.top_k]]
        self.tfidf_scores = {term: score for term, score in sorted_terms[: self.top_k]}

        if embedder is not None and self.vocab:
            self.topic_vectors = self._embed_vocab(embedder)

        return self

    def _load_keyword_scores(self, user_id: str) -> dict[str, float]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT keyword, SUM(frequency) as freq
                   FROM keyword_events WHERE user_id = ?
                   GROUP BY keyword""",
                (user_id,)
            ).fetchall()
        if not rows:
            return {}
        total = sum(r[1] for r in rows)
        return {r[0]: r[1] / total for r in rows}

    def _load_abstract_scores(self, user_id: str) -> dict[str, float]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT paper_abstract, read_seconds
                   FROM paper_interactions
                   WHERE user_id = ? AND paper_abstract != ''""",
                (user_id,)
            ).fetchall()
        if not rows:
            return {}

        doc_term_freq: list[dict[str, int]] = []
        for abstract, read_seconds in rows:
            tokens = _tokenize(abstract)
            freq: dict[str, int] = defaultdict(int)
            for t in tokens:
                freq[t] += 1
            read_weight = min(math.log1p(read_seconds), 3.0)
            doc_term_freq.append((freq, read_weight))

        n_docs = len(doc_term_freq)
        doc_count: dict[str, int] = defaultdict(int)
        for freq, _ in doc_term_freq:
            for term in freq:
                doc_count[term] += 1

        tfidf: dict[str, float] = defaultdict(float)
        for freq, read_weight in doc_term_freq:
            total_tokens = max(sum(freq.values()), 1)
            for term, count in freq.items():
                tf = count / total_tokens
                idf = math.log((n_docs + 1) / (doc_count[term] + 1)) + 1.0
                tfidf[term] += tf * idf * read_weight
        if tfidf:
            total_tfidf = sum(tfidf.values())
            if total_tfidf > 0:
                tfidf = {term: score / total_tfidf for term, score in tfidf.items()}
        return dict(tfidf)

    def _embed_vocab(self, embedder) -> torch.Tensor:
        phrase_inputs = [term for term in self.vocab]

        raw = embedder.model.encode(
            phrase_inputs,
            convert_to_tensor=True,
            show_progress_bar=False,
            device = next(embedder.model.parameters()).device
        )
        normalized = torch.stack([l2_normalize(v) for v in raw])
        return normalized

    def get_weighted_user_vector(self) -> Optional[torch.Tensor]:
        """
        Returns a single 384-dim interest vector as weighted sum of topic vectors.
        Used as the query vector for paper relevance scoring.
        """
        if self.topic_vectors is None or not self.vocab:
            return None
        device = self.topic_vectors.device
        weights = torch.tensor(
             
            [self.tfidf_scores.get(t, 0.0) for t in self.vocab],
            dtype=torch.float32,
            device=device
        )
        
        weights = weights / (weights.sum() + 1e-8)
        weighted = (self.topic_vectors * weights.unsqueeze(1)).sum(dim=0)
        return l2_normalize(weighted)

    def summary(self) -> dict:
        return {
            "vocab_size": len(self.vocab),
            "top_10_topics": self.vocab[:10],
            "has_vectors": self.topic_vectors is not None,
        }
  