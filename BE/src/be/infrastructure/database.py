import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

from be.config import settings


@dataclass(frozen=True)
class UserSampleRecord:
    id: str
    filename: str
    thumb_filename: str
    category: str
    caption: str
    x: float
    y: float
    z: float
    created_at: str


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_samples (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            thumb_filename TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT '',
            caption TEXT NOT NULL DEFAULT '',
            x REAL NOT NULL,
            y REAL NOT NULL,
            z REAL NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_user_samples_created_at
        ON user_samples(created_at DESC);
        """
    )


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    db_missing = not settings.USER_DB_PATH.exists()
    settings.USER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.USER_DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        if db_missing:
            _ensure_schema(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        _ensure_schema(conn)


def insert_user_sample(
    sample_id: str,
    filename: str,
    thumb_filename: str,
    x: float,
    y: float,
    z: float,
    category: str = "",
    caption: str = "",
) -> UserSampleRecord:
    created_at = datetime.now(UTC).isoformat()
    record = UserSampleRecord(
        id=sample_id,
        filename=filename,
        thumb_filename=thumb_filename,
        category=category.strip(),
        caption=caption.strip(),
        x=x,
        y=y,
        z=z,
        created_at=created_at,
    )
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO user_samples (
                id, filename, thumb_filename, category, caption, x, y, z, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                record.id,
                record.filename,
                record.thumb_filename,
                record.category,
                record.caption,
                record.x,
                record.y,
                record.z,
                record.created_at,
            ),
        )
    return record


def list_user_samples() -> list[UserSampleRecord]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, filename, thumb_filename, category, caption, x, y, z, created_at
            FROM user_samples
            ORDER BY created_at DESC;
            """
        ).fetchall()

    return [
        UserSampleRecord(
            id=str(row["id"]),
            filename=str(row["filename"]),
            thumb_filename=str(row["thumb_filename"]),
            category=str(row["category"]),
            caption=str(row["caption"]),
            x=float(row["x"]),
            y=float(row["y"]),
            z=float(row["z"]),
            created_at=str(row["created_at"]),
        )
        for row in rows
    ]
