import colorsys
import json
import os
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from be.config import settings

_VIDEO_ID_RE = re.compile(r"^(.+?)_+frame", re.IGNORECASE)
_FTS_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class SampleRecord:
    idx: int
    sample_idx: str
    image_path: str
    video_id: str
    category: str
    caption: str
    x: float
    y: float
    z: float
    cluster: int
    atlas_id: int
    tile_index: int


def extract_video_id(image_path: str) -> str:
    """Wyciąga identyfikator wideo YouTube z nazwy pliku (np. p0I7ccItmv8_frame_393482.jpg)."""
    stem = Path(image_path).stem
    match = _VIDEO_ID_RE.match(stem)
    return match.group(1) if match else ""


def build_clusters_metadata(labels: np.ndarray, n_clusters: int) -> list[dict]:
    """Generuje listę klastrów z unikalnymi kolorami HSV i liczebnością."""
    counts = np.bincount(labels.astype(np.int64, copy=False), minlength=n_clusters)
    result = []
    for i in range(n_clusters):
        r, g, b = colorsys.hsv_to_rgb(i / max(n_clusters, 1), 0.75, 0.95)
        color_hex = f"#{int(255 * r):02x}{int(255 * g):02x}{int(255 * b):02x}"
        result.append({"id": i, "color": color_hex, "count": int(counts[i])})
    return result


def save_clusters_json(clusters: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(clusters, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def build_metadata_db(
    db_path: Path,
    sample_ids: Sequence[str],
    image_paths: Sequence[str],
    categories: Sequence[str],
    captions: Sequence[str],
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    clusters: np.ndarray,
    atlas_ids: np.ndarray | None = None,
    tile_indices: np.ndarray | None = None,
) -> None:
    """Tworzy atomowo bazę SQLite z pełnymi metadanymi punktów oraz indeksem FTS5."""
    n = len(image_paths)
    if atlas_ids is None:
        atlas_ids = np.full(n, -1, dtype=np.int32)
    if tile_indices is None:
        tile_indices = np.full(n, -1, dtype=np.int32)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(".db.tmp")
    tmp_path.unlink(missing_ok=True)

    conn = sqlite3.connect(tmp_path)
    try:
        conn.execute("PRAGMA journal_mode=OFF;")
        conn.execute("PRAGMA synchronous=OFF;")
        conn.execute("PRAGMA temp_store=MEMORY;")

        conn.execute(
            """
            CREATE TABLE samples (
                idx INTEGER PRIMARY KEY,
                sample_idx TEXT NOT NULL,
                image_path TEXT NOT NULL,
                video_id TEXT NOT NULL,
                category TEXT NOT NULL,
                caption TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                cluster INTEGER NOT NULL,
                atlas_id INTEGER NOT NULL,
                tile_index INTEGER NOT NULL
            );
            """
        )

        rows = (
            (
                i,
                str(sample_ids[i]),
                str(image_paths[i]),
                extract_video_id(str(image_paths[i])),
                str(categories[i]),
                str(captions[i]),
                float(xs[i]),
                float(ys[i]),
                float(zs[i]),
                int(clusters[i]),
                int(atlas_ids[i]),
                int(tile_indices[i]),
            )
            for i in range(n)
        )
        conn.executemany(
            """
            INSERT INTO samples (
                idx, sample_idx, image_path, video_id, category, caption,
                x, y, z, cluster, atlas_id, tile_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )

        conn.execute("CREATE INDEX idx_samples_cluster ON samples(cluster);")
        conn.execute("CREATE INDEX idx_samples_video_id ON samples(video_id);")

        conn.execute(
            """
            CREATE VIRTUAL TABLE samples_fts USING fts5(
                caption,
                category,
                video_id,
                image_path,
                content='samples',
                content_rowid='idx',
                tokenize='unicode61'
            );
            """
        )
        conn.execute("INSERT INTO samples_fts(samples_fts) VALUES('rebuild');")
        conn.commit()
    finally:
        conn.close()

    os.replace(tmp_path, db_path)


def update_atlas_mappings(
    db_path: Path, atlas_ids: np.ndarray, tile_indices: np.ndarray
) -> None:
    """Aktualizuje kolumny atlas_id oraz tile_index w istniejącej bazie metadata.db."""
    if not db_path.exists():
        return

    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.executemany(
            "UPDATE samples SET atlas_id = ?, tile_index = ? WHERE idx = ?;",
            (
                (int(atlas_ids[i]), int(tile_indices[i]), i)
                for i in range(len(atlas_ids))
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _open_readonly_connection() -> sqlite3.Connection:
    db_path = settings.METADATA_DB_PATH
    if not db_path.exists():
        raise FileNotFoundError(f"Brak bazy metadanych: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_sample(row: sqlite3.Row) -> SampleRecord:
    return SampleRecord(
        idx=int(row["idx"]),
        sample_idx=str(row["sample_idx"]),
        image_path=str(row["image_path"]),
        video_id=str(row["video_id"]),
        category=str(row["category"]),
        caption=str(row["caption"]),
        x=float(row["x"]),
        y=float(row["y"]),
        z=float(row["z"]),
        cluster=int(row["cluster"]),
        atlas_id=int(row["atlas_id"]),
        tile_index=int(row["tile_index"]),
    )


def get_image_paths_ordered(db_path: Path | None = None) -> list[str]:
    """Zwraca uporządkowaną listę image_path dla wszystkich punktów (po idx)."""
    target = db_path or settings.METADATA_DB_PATH
    conn = sqlite3.connect(f"file:{target}?mode=ro", uri=True, timeout=10.0)
    try:
        rows = conn.execute(
            "SELECT image_path FROM samples ORDER BY idx ASC;"
        ).fetchall()
        return [str(r[0]) for r in rows]
    finally:
        conn.close()


def get_sample_by_idx(idx: int) -> SampleRecord | None:
    conn = _open_readonly_connection()
    try:
        row = conn.execute(
            """
            SELECT idx, sample_idx, image_path, video_id, category, caption,
                   x, y, z, cluster, atlas_id, tile_index
            FROM samples
            WHERE idx = ?;
            """,
            (idx,),
        ).fetchone()
        return _row_to_sample(row) if row is not None else None
    finally:
        conn.close()


def _build_fts_query(raw_query: str) -> str:
    tokens = _FTS_TOKEN_RE.findall(raw_query.strip())
    if not tokens:
        return ""
    return " AND ".join(f'"{tok}"*' for tok in tokens[:12])


def search_samples_fts(
    query: str,
    limit: int = 50,
    offset: int = 0,
    cluster: int | None = None,
) -> list[SampleRecord]:
    fts_query = _build_fts_query(query)
    if not fts_query:
        return []

    conn = _open_readonly_connection()
    try:
        if cluster is None:
            rows = conn.execute(
                """
                SELECT s.idx, s.sample_idx, s.image_path, s.video_id, s.category,
                       s.caption, s.x, s.y, s.z, s.cluster, s.atlas_id, s.tile_index
                FROM samples_fts f
                JOIN samples s ON s.idx = f.rowid
                WHERE samples_fts MATCH ?
                ORDER BY bm25(samples_fts)
                LIMIT ? OFFSET ?;
                """,
                (fts_query, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT s.idx, s.sample_idx, s.image_path, s.video_id, s.category,
                       s.caption, s.x, s.y, s.z, s.cluster, s.atlas_id, s.tile_index
                FROM samples_fts f
                JOIN samples s ON s.idx = f.rowid
                WHERE samples_fts MATCH ? AND s.cluster = ?
                ORDER BY bm25(samples_fts)
                LIMIT ? OFFSET ?;
                """,
                (fts_query, cluster, limit, offset),
            ).fetchall()
        return [_row_to_sample(row) for row in rows]
    finally:
        conn.close()


def get_dataset_bounds() -> tuple[int, dict[str, list[float]]]:
    conn = _open_readonly_connection()
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS cnt,
                   MIN(x) AS min_x, MAX(x) AS max_x,
                   MIN(y) AS min_y, MAX(y) AS max_y,
                   MIN(z) AS min_z, MAX(z) AS max_z
            FROM samples;
            """
        ).fetchone()
        cnt = int(row["cnt"])
        if cnt == 0:
            return 0, {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
        return cnt, {
            "min": [float(row["min_x"]), float(row["min_y"]), float(row["min_z"])],
            "max": [float(row["max_x"]), float(row["max_y"]), float(row["max_z"])],
        }
    finally:
        conn.close()
