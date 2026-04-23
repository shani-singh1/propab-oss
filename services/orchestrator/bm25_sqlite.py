from __future__ import annotations

import sqlite3
from pathlib import Path


def _db_path(session_id: str, data_dir: str) -> Path:
    path = Path(data_dir) / "bm25" / f"{session_id}.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_session_chunks(session_id: str, data_dir: str, rows: list[tuple[str, int, str]]) -> None:
    path = _db_path(session_id, data_dir)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk (
                paper_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                PRIMARY KEY (paper_id, chunk_index)
            )
            """
        )
        conn.execute("DELETE FROM chunk")
        conn.executemany("INSERT INTO chunk (paper_id, chunk_index, text) VALUES (?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()


def read_session_chunks(session_id: str, data_dir: str) -> list[tuple[str, int, str]]:
    path = _db_path(session_id, data_dir)
    if not path.exists():
        return []
    conn = sqlite3.connect(path)
    try:
        cur = conn.execute("SELECT paper_id, chunk_index, text FROM chunk ORDER BY paper_id, chunk_index")
        return [(str(r[0]), int(r[1]), str(r[2])) for r in cur.fetchall()]
    finally:
        conn.close()
