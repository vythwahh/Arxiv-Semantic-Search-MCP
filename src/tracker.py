import sqlite3
import uuid
import time
import re
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Optional
 

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "behavior_events.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        schema = SCHEMA_PATH.read_text()
        conn.executescript(schema)
        conn.commit()


@contextmanager
def get_conn(db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_user(user_id: str, email: Optional[str] = None, db_path: Path = DB_PATH) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO users (user_id, email) VALUES (?, ?)",
            (user_id, email)
        )
        if email:
            conn.execute(
                "UPDATE users SET email = ? WHERE user_id = ? AND email IS NULL",
                (email, user_id)
            )


class SessionTracker:
    def __init__(self, user_id: str, query: str, db_path: Path = DB_PATH):
        self.user_id = user_id
        self.query = query
        self.db_path = db_path
        self.session_id = str(uuid.uuid4())
        self.start_time = time.monotonic()
        self.scroll_start: Optional[float] = None
        self.scroll_accumulated: float = 0.0
        self.result_count: int = 0

    def start(self, result_count: int = 0) -> str:
        self.result_count = result_count
        with get_conn(self.db_path) as conn:
            conn.execute(
                """INSERT INTO search_sessions
                   (session_id, user_id, query, started_at, result_count)
                   VALUES (?, ?, ?, ?, ?)""",
                (self.session_id, self.user_id, self.query,
                 datetime.utcnow().isoformat(), result_count)
            )
        self._record_keywords()
        return self.session_id

    def _record_keywords(self) -> None:
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', self.query.lower())
        stopwords = {
            'the', 'and', 'for', 'are', 'with', 'that', 'this',
            'from', 'how', 'what', 'when', 'where', 'which'
        }
        keywords = [t for t in tokens if t not in stopwords]
        if not keywords:
            return
        freq: dict[str, int] = {}
        for kw in keywords:
            freq[kw] = freq.get(kw, 0) + 1
        with get_conn(self.db_path) as conn:
            conn.executemany(
                """INSERT INTO keyword_events
                   (user_id, keyword, session_id, frequency, recorded_at)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (self.user_id, kw, self.session_id, count,
                     datetime.utcnow().isoformat())
                    for kw, count in freq.items()
                ]
            )

    def begin_scroll(self) -> None:
        if self.scroll_start is None:
            self.scroll_start = time.monotonic()

    def end_scroll(self) -> None:
        if self.scroll_start is not None:
            self.scroll_accumulated += time.monotonic() - self.scroll_start
            self.scroll_start = None

    def end(self) -> dict:
        if self.scroll_start is not None:
            self.end_scroll()
        dwell = time.monotonic() - self.start_time
        with get_conn(self.db_path) as conn:
            conn.execute(
                """UPDATE search_sessions
                   SET ended_at = ?, dwell_seconds = ?, scroll_seconds = ?
                   WHERE session_id = ?""",
                (datetime.utcnow().isoformat(), round(dwell, 3),
                 round(self.scroll_accumulated, 3), self.session_id)
            )
        return {
            "session_id": self.session_id,
            "dwell_seconds": round(dwell, 3),
            "scroll_seconds": round(self.scroll_accumulated, 3),
        }


class PaperReadTracker:
    def __init__(self, user_id: str, session_id: str, paper_id: str,
                 paper_title: str = "", paper_abstract: str = "",
                 paper_categories: str = "", db_path: Path = DB_PATH):
        self.user_id = user_id
        self.session_id = session_id
        self.paper_id = paper_id
        self.paper_title = paper_title
        self.paper_abstract = paper_abstract
        self.paper_categories = paper_categories
        self.db_path = db_path
        self.click_time = datetime.utcnow().isoformat()
        self.read_start = time.monotonic()

    def start(self) -> None:
        with get_conn(self.db_path) as conn:
            conn.execute(
                """INSERT INTO paper_interactions
                   (session_id, user_id, paper_id, paper_title,
                    paper_abstract, paper_categories, clicked_at, read_seconds)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
                (self.session_id, self.user_id, self.paper_id,
                 self.paper_title, self.paper_abstract,
                 self.paper_categories, self.click_time)
            )

    def end(self) -> float:
        read_seconds = round(time.monotonic() - self.read_start, 3)
        with get_conn(self.db_path) as conn:
            conn.execute(
                """UPDATE paper_interactions
                   SET read_seconds = ?
                   WHERE session_id = ? AND paper_id = ? AND clicked_at = ?""",
                (read_seconds, self.session_id, self.paper_id, self.click_time)
            )
        return read_seconds


def get_user_behavior_summary(user_id: str, db_path: Path = DB_PATH) -> dict:
    with get_conn(db_path) as conn:
        sessions = conn.execute(
            """SELECT COUNT(*) as total_sessions,
                      AVG(dwell_seconds) as avg_dwell,
                      AVG(scroll_seconds) as avg_scroll
               FROM search_sessions WHERE user_id = ?""",
            (user_id,)
        ).fetchone()

        interactions = conn.execute(
            """SELECT COUNT(*) as total_clicks,
                      AVG(read_seconds) as avg_read
               FROM paper_interactions WHERE user_id = ?""",
            (user_id,)
        ).fetchone()

        top_keywords = conn.execute(
            """SELECT keyword, SUM(frequency) as total_freq
               FROM keyword_events WHERE user_id = ?
               GROUP BY keyword ORDER BY total_freq DESC LIMIT 20""",
            (user_id,)
        ).fetchall()

    return {
        "user_id": user_id,
        "total_sessions": sessions["total_sessions"] or 0,
        "avg_dwell_seconds": round(sessions["avg_dwell"] or 0, 2),
        "avg_scroll_seconds": round(sessions["avg_scroll"] or 0, 2),
        "total_clicks": interactions["total_clicks"] or 0,
        "avg_read_seconds": round(interactions["avg_read"] or 0, 2),
        "top_keywords": [
            {"keyword": r["keyword"], "frequency": r["total_freq"]}
            for r in top_keywords
        ],
    }
