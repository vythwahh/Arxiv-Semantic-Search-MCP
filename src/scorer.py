import sqlite3
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from embedder import ArxivEmbedder, Paper
from interest_model import InterestModel, DB_PATH, MODEL_PATH
from vector_ops import l2_normalize


@dataclass
class ScoredPaper:
    paper: Paper
    score: float
    user_id: str


def load_user_model(
    user_id: str,
    db_path: Path = DB_PATH,
    model_path: Path = MODEL_PATH,
) -> Optional[InterestModel]:
    user_model_path = model_path.parent / f"interest_model_{user_id}.pt"
    model = InterestModel(user_id=user_id, db_path=db_path, model_path=user_model_path)
    loaded = model.load()
    if not loaded:
        return None
    return model


def score_papers_for_user(
    papers: list[Paper],
    user_id: str,
    embedder: ArxivEmbedder,
    db_path: Path = DB_PATH,
    model_path: Path = MODEL_PATH,
    top_n: int = 5,
    threshold: float = 0.6,
) -> list[ScoredPaper]:
    """
    Scores a list of papers against a user's interest model.
    Returns top-N papers that are also above the score threshold.

    Args:
        papers: List of Paper objects fetched from ArXiv.
        user_id: Target user to score against.
        embedder: ArxivEmbedder instance for generating paper embeddings.
        top_n: Maximum number of papers to return.
        threshold: Minimum interest score [0, 1] to qualify for notification.

    Returns:
        List of ScoredPaper sorted by score descending, filtered by threshold and top_n.
    """
    model = load_user_model(user_id, db_path=db_path, model_path=model_path)
    if model is None:
        return []

    scored: list[ScoredPaper] = []
    for paper in papers:
        text = f"{paper.title}. {paper.abstract}"
        embedding = embedder.model.encode(
            text,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        score = model.score_paper(embedding)
        if score >= threshold:
            scored.append(ScoredPaper(paper=paper, score=score, user_id=user_id))

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_n]


def score_papers_all_users(
    papers: list[Paper],
    embedder: ArxivEmbedder,
    db_path: Path = DB_PATH,
    model_path: Path = MODEL_PATH,
    top_n: int = 5,
    threshold: float = 0.6,
) -> dict[str, list[ScoredPaper]]:
    """
    Scores papers for all users in the DB.
    Returns a dict mapping user_id -> list of ScoredPaper.
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT user_id FROM users").fetchall()
    user_ids = [r[0] for r in rows]

    results: dict[str, list[ScoredPaper]] = {}
    for uid in user_ids:
        scored = score_papers_for_user(
            papers=papers,
            user_id=uid,
            embedder=embedder,
            db_path=db_path,
            model_path=model_path,
            top_n=top_n,
            threshold=threshold,
        )
        if scored:
            results[uid] = scored

    return results
