import math

import torch
import torch.nn as nn
import sqlite3
from pathlib import Path
from typing import Optional
from vector_ops import l2_normalize
from user_profile import UserBehaviorProfile
from topic_builder import TopicVocab


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "behavior_events.db"
MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "interest_model.pt"

SIGNAL_WEIGHTS = {
    "keyword": 0.2,
    "scroll":  0.3,
    "click":   0.5,
    "read":    1.0,
}


class BehaviorSignalEncoder(nn.Module):
    """
    Encodes raw behavior scalars into a normalized interest score [0, 1].

    Input: 4-dim vector [dwell_norm, scroll_norm, click_norm, read_norm]
    Output: scalar interest score via learned linear combination + sigmoid

    Trained with BCELoss against implicit positive labels (any interaction = 1).
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)
        nn.init.constant_(self.fc.weight, 0.25)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))


class InterestModel:
    """
    Full interest modeling pipeline for a single user.

    Combines:
      - BehaviorSignalEncoder: learns scalar interest score per session
      - UserBehaviorProfile: maintains EMA interest vector + K-Means centroids
      - TopicVocab: maps behavior history to topic vocabulary + embeddings

    Usage:
        model = InterestModel(user_id="user_001")
        model.ingest_from_db(embedder)
        score = model.score_paper(paper_embedding)
        model.save()
    """

    def __init__(self, user_id: str, db_path: Path = DB_PATH,
                 model_path: Path = MODEL_PATH):
        self.user_id = user_id
        self.db_path = db_path
        self.model_path = model_path

        self.encoder = BehaviorSignalEncoder()
        self.profile = UserBehaviorProfile(user_id=user_id)
        self.topic_vocab: Optional[TopicVocab] = None
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        self.loss_fn = nn.BCELoss()

    def _load_sessions(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT s.session_id, s.dwell_seconds, s.scroll_seconds,
                          COUNT(p.interaction_id) as click_count,
                          COALESCE(SUM(p.read_seconds), 0) as total_read
                   FROM search_sessions s
                   LEFT JOIN paper_interactions p
                     ON s.session_id = p.session_id
                   WHERE s.user_id = ?
                   GROUP BY s.session_id""",
                (self.user_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def _normalize_signals(self, sessions: list[dict]) -> torch.Tensor:
        """
        Normalizes raw session signals to [0, 1] using per-column max scaling.
        Returns tensor of shape (n_sessions, 4).
        """
        if not sessions:
            return torch.empty(0, 4)

        raw = torch.tensor(
            [[s["dwell_seconds"], s["scroll_seconds"],
              float(s["click_count"]), s["total_read"]]
             for s in sessions],
            dtype=torch.float32,
        )
        col_max = raw.max(dim=0).values.clamp(min=1.0)
        return raw / col_max

    def train_encoder(self, n_epochs: int = 20) -> list[float]:
        """
        Trains BehaviorSignalEncoder on implicit positive labels.
        Any session with at least one click is labeled 1.0, else 0.0.
        """
        sessions = self._load_sessions()
        if not sessions:
            return []

        X = self._normalize_signals(sessions)
        y = torch.tensor(
            [1.0 if s["click_count"] > 0 else 0.0 for s in sessions],
            dtype=torch.float32,
        ).unsqueeze(1)

        losses = []
        self.encoder.train()
        for _ in range(n_epochs):
            self.optimizer.zero_grad()
            pred = self.encoder(X)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            losses.append(round(loss.item(), 6))

        return losses

    def ingest_from_db(self, embedder=None) -> None:
        """
        Reads all behavior events from DB, feeds them into UserBehaviorProfile
        using combined session and interaction weights, and builds TopicVocab.
        """
        sessions = self._load_sessions()
        if not sessions:
            return
            
        if embedder is not None:
            X = self._normalize_signals(sessions)
            self.encoder.eval()
            with torch.no_grad():
                scores = self.encoder(X).squeeze(1)
                
             
            session_scores = {
                s["session_id"]: scores[i].item()
                for i, s in enumerate(sessions)
            }
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                interactions = conn.execute(
                    """SELECT session_id, paper_abstract, paper_categories, read_seconds
                       FROM paper_interactions
                       WHERE user_id = ? AND paper_abstract != ''""",
                    (self.user_id,)
                ).fetchall()
                
            for row in interactions:
                text = f"{row['paper_categories']} {row['paper_abstract']}"
                embedding = embedder.model.encode(
                    text,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                
 
                interest_score = session_scores.get(row["session_id"], 0.5)
                
                
                read_weight = min(math.log1p(row["read_seconds"]) * SIGNAL_WEIGHTS["read"], 2.0)
                
 
                combined_weight = interest_score * read_weight
                
 
                self.profile.log_action(embedding, weight=combined_weight)
                
        self.topic_vocab = TopicVocab(db_path=self.db_path)
        self.topic_vocab.build(self.user_id, embedder=embedder)

    def score_paper(self, paper_embedding: torch.Tensor) -> float:
        """
        Scores a paper embedding against the user's interest profile.

        Returns a float in [0, 1] — higher = more relevant to user.
        Uses:
          - UserBehaviorProfile.global_profile_tensor (EMA interest vector)
          - TopicVocab.get_weighted_user_vector() (TF-IDF topic vector)
          Both averaged if available.
        """
        paper_vec = l2_normalize(paper_embedding.flatten().float())
        scores = []

        if self.profile.global_profile_tensor is not None:
            global_vec = self.profile.global_profile_tensor.to(paper_vec.device).float()
            sim = torch.dot(paper_vec, global_vec).item()
            scores.append((sim + 1.0) / 2.0)

        if self.topic_vocab is not None:
            user_topic_vec = self.topic_vocab.get_weighted_user_vector()
            if user_topic_vec is not None:
                user_topic_vec = user_topic_vec.to(paper_vec.device).float()
                sim = torch.dot(paper_vec, user_topic_vec).item()
                scores.append((sim + 1.0) / 2.0)

        if not scores:
            return 0.0
        return round(sum(scores) / len(scores), 4)

    def get_top_interests(self) -> list[str]:
        if self.topic_vocab is None:
            return []
        return self.topic_vocab.vocab[:10]

    def save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "encoder_state": self.encoder.state_dict(),
            "global_profile": self.profile.global_profile_tensor,
            "interest_centroids": self.profile.interest_centroids,
            "vocab": self.topic_vocab.vocab if self.topic_vocab else [],
            "tfidf_scores": self.topic_vocab.tfidf_scores if self.topic_vocab else {},
        }, self.model_path)

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        checkpoint = torch.load(self.model_path, weights_only=False)
        self.encoder.load_state_dict(checkpoint["encoder_state"])
        self.profile.global_profile_tensor = checkpoint["global_profile"]
        self.profile.interest_centroids = checkpoint["interest_centroids"]
        if checkpoint["vocab"]:
            self.topic_vocab = TopicVocab(db_path=self.db_path)
            self.topic_vocab.vocab = checkpoint["vocab"]
            self.topic_vocab.tfidf_scores = checkpoint["tfidf_scores"]
        return True
