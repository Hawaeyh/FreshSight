"""Minimal local SQLite persistence for analyses and user verification."""

from __future__ import annotations

import sqlite3
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ai.dataset import CLASS_NAMES
from config.paths import DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_uuid TEXT NOT NULL UNIQUE,
    stored_image_path TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    predicted_class TEXT,
    confidence REAL,
    fresh_probability REAL,
    unripe_probability REAL,
    rotten_probability REAL,
    matlab_rule_class TEXT,
    damage_percentage REAL,
    healthy_percentage REAL,
    model_version TEXT,
    analyzed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prediction_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    feedback_status TEXT NOT NULL,
    predicted_class TEXT,
    corrected_class TEXT,
    submitted_at TEXT NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'pending',
    included_in_retraining INTEGER NOT NULL DEFAULT 0,
    image_sha256 TEXT,
    FOREIGN KEY (analysis_id) REFERENCES analysis_history(id)
);
CREATE TABLE IF NOT EXISTS active_learning_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL UNIQUE,
    candidate_score REAL NOT NULL,
    reasons_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    FOREIGN KEY (analysis_id) REFERENCES analysis_history(id)
);
"""


class HistoryService:
    def __init__(self, database_path: Path = DB_PATH):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self):
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            existing = {row[1] for row in connection.execute("PRAGMA table_info(analysis_history)")}
            additions = {
                "image_sha256": "TEXT", "confidence_level": "TEXT",
                "top_two_probability_margin": "REAL", "inference_device": "TEXT",
                "ai_processing_time": "REAL", "matlab_processing_time": "REAL",
                "ai_matlab_agreement": "INTEGER", "requires_manual_review": "INTEGER",
            }
            for name, sql_type in additions.items():
                if name not in existing:
                    connection.execute(f"ALTER TABLE analysis_history ADD COLUMN {name} {sql_type}")
            feedback_columns = {row[1] for row in connection.execute("PRAGMA table_info(prediction_feedback)")}
            if "image_sha256" not in feedback_columns:
                connection.execute("ALTER TABLE prediction_feedback ADD COLUMN image_sha256 TEXT")
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_image_sha256 "
                "ON prediction_feedback(image_sha256) WHERE image_sha256 IS NOT NULL"
            )
            # Add content identities to pre-migration rows without changing source images.
            for row in connection.execute(
                "SELECT id, stored_image_path FROM analysis_history WHERE image_sha256 IS NULL"
            ).fetchall():
                source = Path(row["stored_image_path"])
                if source.is_file():
                    digest = hashlib.sha256(source.read_bytes()).hexdigest()
                    connection.execute(
                        "UPDATE analysis_history SET image_sha256=? WHERE id=?", (digest, row["id"])
                    )
            for row in connection.execute(
                """SELECT f.id, a.image_sha256 FROM prediction_feedback f
                   JOIN analysis_history a ON a.id=f.analysis_id
                   WHERE f.image_sha256 IS NULL AND a.image_sha256 IS NOT NULL"""
            ).fetchall():
                already_used = connection.execute(
                    "SELECT 1 FROM prediction_feedback WHERE image_sha256=?", (row["image_sha256"],)
                ).fetchone()
                if not already_used:
                    connection.execute(
                        "UPDATE prediction_feedback SET image_sha256=? WHERE id=?",
                        (row["image_sha256"], row["id"]),
                    )

    def record_analysis(self, analysis_uuid: str, stored_image_path: str,
                        original_filename: str, result: dict) -> int:
        ai = result.get("ai_detection") or {}
        matlab = result.get("matlab_analysis") or {}
        probabilities = ai.get("probabilities") or {}
        measurements = matlab.get("measurements") or {}
        assessment = result.get("system_assessment") or {}
        source = Path(stored_image_path)
        image_sha256 = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
        values = (
            analysis_uuid, stored_image_path, original_filename,
            ai.get("predicted_class"), ai.get("confidence"),
            probabilities.get("Fresh"), probabilities.get("Unripe"),
            probabilities.get("Rotten"), matlab.get("rule_class"),
            matlab.get("damage_percentage", measurements.get("damage_percentage")),
            matlab.get("healthy_percentage", measurements.get("healthy_percentage")),
            ai.get("model_version"), datetime.now(timezone.utc).isoformat(),
            image_sha256, ai.get("confidence_level"), ai.get("top_two_probability_margin"),
            ai.get("device"), ai.get("processing_time_seconds"),
            matlab.get("processing_time_seconds"),
            None if assessment.get("ai_matlab_agreement") is None else int(assessment["ai_matlab_agreement"]),
            int(bool(assessment.get("requires_manual_review"))),
        )
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO analysis_history (
                    analysis_uuid, stored_image_path, original_filename, predicted_class,
                    confidence, fresh_probability, unripe_probability, rotten_probability,
                    matlab_rule_class, damage_percentage, healthy_percentage, model_version,
                    analyzed_at, image_sha256, confidence_level, top_two_probability_margin,
                    inference_device, ai_processing_time, matlab_processing_time,
                    ai_matlab_agreement, requires_manual_review
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", values,
            )
            return int(cursor.lastrowid)

    def record_feedback(self, analysis_uuid: str, feedback_status: str,
                        corrected_class: str | None = None) -> dict:
        if feedback_status not in {"correct", "incorrect"}:
            raise ValueError("feedback_status must be 'correct' or 'incorrect'.")
        if feedback_status == "incorrect" and corrected_class not in CLASS_NAMES:
            raise ValueError(f"corrected_class must be one of: {', '.join(CLASS_NAMES)}")
        if feedback_status == "correct":
            corrected_class = None
        with self._connect() as connection:
            analysis = connection.execute(
                "SELECT id, predicted_class, image_sha256 FROM analysis_history WHERE analysis_uuid = ?",
                (analysis_uuid,),
            ).fetchone()
            if analysis is None:
                raise LookupError("Analysis ID was not found.")
            if analysis["image_sha256"]:
                duplicate = connection.execute(
                    "SELECT id FROM prediction_feedback WHERE image_sha256 = ?",
                    (analysis["image_sha256"],),
                ).fetchone()
                if duplicate:
                    raise ValueError("Feedback already exists for identical image content.")
            cursor = connection.execute(
                """INSERT INTO prediction_feedback (
                    analysis_id, feedback_status, predicted_class, corrected_class,
                    submitted_at, review_status, included_in_retraining, image_sha256
                ) VALUES (?, ?, ?, ?, ?, 'pending', 0, ?)""",
                (analysis["id"], feedback_status, analysis["predicted_class"], corrected_class,
                 datetime.now(timezone.utc).isoformat(), analysis["image_sha256"]),
            )
            return {
                "id": int(cursor.lastrowid), "analysis_id": int(analysis["id"]),
                "feedback_status": feedback_status,
                "predicted_class": analysis["predicted_class"],
                "corrected_class": corrected_class, "review_status": "pending",
                "included_in_retraining": False,
            }

    def get_analysis(self, analysis_uuid: str):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM analysis_history WHERE analysis_uuid = ?", (analysis_uuid,)
            ).fetchone()
            return dict(row) if row else None

    def list_analyses(self, limit: int = 100):
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT * FROM analysis_history ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()]

    def list_feedback(self):
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(
                """SELECT f.*, a.analysis_uuid, a.original_filename, a.confidence
                   FROM prediction_feedback f JOIN analysis_history a ON a.id=f.analysis_id
                   ORDER BY f.id DESC"""
            ).fetchall()]

    def review_feedback(self, feedback_id: int, status: str):
        if status not in {"approved", "rejected"}:
            raise ValueError("review_status must be approved or rejected.")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE prediction_feedback SET review_status=? WHERE id=?", (status, feedback_id)
            )
            if cursor.rowcount != 1:
                raise LookupError("Feedback record was not found.")

    def save_candidate(self, analysis_id: int, score: float, reasons: list[str]):
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO active_learning_candidates
                   (analysis_id,candidate_score,reasons_json,status,created_at)
                   VALUES(?,?,?,'pending',?)
                   ON CONFLICT(analysis_id) DO UPDATE SET
                   candidate_score=excluded.candidate_score,reasons_json=excluded.reasons_json""",
                (analysis_id, float(score), json.dumps(reasons), datetime.now(timezone.utc).isoformat()),
            )

    def list_candidates(self):
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(
                """SELECT c.*,a.analysis_uuid,a.original_filename,a.predicted_class,a.confidence
                   FROM active_learning_candidates c JOIN analysis_history a ON a.id=c.analysis_id
                   ORDER BY c.candidate_score DESC"""
            ).fetchall()]


_SERVICE = None


def get_history_service() -> HistoryService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = HistoryService()
    return _SERVICE
