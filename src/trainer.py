import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from pytz import utc
from interest_model import InterestModel, DB_PATH, MODEL_PATH


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trainer")


def get_all_user_ids(db_path: Path = DB_PATH) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT user_id FROM users").fetchall()
    return [r[0] for r in rows]


def retrain_user(
    user_id: str,
    embedder=None,
    db_path: Path = DB_PATH,
    model_path: Path = MODEL_PATH,
    n_epochs: int = 20,
) -> dict:
    """
    Retrains InterestModel for a single user.
    Returns a summary dict with user_id, loss trajectory, and timestamp.
    """
    logger.info(f"Retraining model for user: {user_id}")
    
    model = InterestModel(user_id=user_id, db_path=db_path, model_path=model_path)

    loaded = model.load()
    if loaded:
        logger.info(f"Loaded existing checkpoint for {user_id}")

    losses = model.train_encoder(n_epochs=n_epochs)
    if not losses:
        logger.warning(f"No sessions found for {user_id}, skipping")
        return {"user_id": user_id, "status": "skipped", "reason": "no sessions"}

    model.ingest_from_db(embedder=embedder)
    model.save()

    summary = {
        "user_id": user_id,
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "n_epochs": n_epochs,
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "top_interests": model.get_top_interests(),
    }
    logger.info(
        f"Done {user_id}: loss {losses[0]:.4f} -> {losses[-1]:.4f} | "
        f"interests: {model.get_top_interests()[:5]}"
    )
    return summary


def retrain_all(
    embedder=None,
    db_path: Path = DB_PATH,
    model_path: Path = MODEL_PATH,
    n_epochs: int = 20,
) -> list[dict]:
    """
    Retrains InterestModel for all users in the DB.
    Called by APScheduler nightly or directly by cron.
    """
    user_ids = get_all_user_ids(db_path)
    if not user_ids:
        logger.warning("No users found in DB, nothing to retrain")
        return []

    logger.info(f"Starting nightly retrain for {len(user_ids)} user(s)")
    results = []
    for uid in user_ids:
        try:
            result = retrain_user(uid, embedder=embedder, db_path=db_path,
                                  model_path=model_path, n_epochs=n_epochs)
            results.append(result)
        except Exception as e:
            logger.error(f"Retrain failed for {uid}: {e}")
            results.append({"user_id": uid, "status": "error", "reason": str(e)})

    ok = sum(1 for r in results if r["status"] == "ok")
    logger.info(f"Nightly retrain complete: {ok}/{len(results)} succeeded")
    return results


class NightlyTrainer:
    """
    Wraps APScheduler to run retrain_all() on a nightly cron schedule.
    Designed to run as a background thread alongside the MCP server.

    Usage:
        trainer = NightlyTrainer(embedder=embedder)
        trainer.start()   # non-blocking, runs in background
        ...
        trainer.stop()
    """

    def __init__(
        self,
        embedder=None,
        db_path: Path = DB_PATH,
        model_path: Path = MODEL_PATH,
        n_epochs: int = 20,
        hour: int = 2,
        minute: int = 0,
    ):
        self.embedder = embedder
        self.db_path = db_path
        self.model_path = model_path
        self.n_epochs = n_epochs
        self.hour = hour
        self.minute = minute
        self._scheduler: Optional[BackgroundScheduler] = None

    def _job(self) -> None:
        retrain_all(
            embedder=self.embedder,
            db_path=self.db_path,
            model_path=self.model_path,
            n_epochs=self.n_epochs,
        )

    def start(self) -> None:
        self._scheduler = BackgroundScheduler(timezone=utc)
        self._scheduler.add_job(
            self._job,
            trigger=CronTrigger(hour=self.hour, minute=self.minute, timezone=utc),
            id="nightly_retrain",
            name="Nightly interest model retrain",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info(
            f"NightlyTrainer started — scheduled at {self.hour:02d}:{self.minute:02d} UTC daily"
        )

    def run_now(self) -> list[dict]:
        """Triggers an immediate retrain outside the schedule. Useful for testing."""
        logger.info("Manual retrain triggered")
        return retrain_all(
            embedder=self.embedder,
            db_path=self.db_path,
            model_path=self.model_path,
            n_epochs=self.n_epochs,
        )

    def stop(self) -> None:
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("NightlyTrainer stopped")


if __name__ == "__main__":
    """
    Cron fallback entry point.
    Add to crontab: 0 2 * * * cd /path/to/project && python src/trainer.py
    """
    import sys
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else DB_PATH
    from embedder import ArxivEmbedder
    embedder = ArxivEmbedder()
    results = retrain_all(embedder=embedder, db_path=db)
    for r in results:
        print(r)
