"""Ground-truth event capture (label store).

Records operational events — panel cleanings, faults, capacity tests, maintenance —
that the telemetry stream does not contain. These labels are the prerequisite for the
supervised ML on the roadmap (root-cause classification, RUL), so they live in their
OWN SQLite file (config.EVENTS_DB_PATH), separate from the telemetry DB that
compile_db.py drops and rebuilds.
"""
import sqlite3
from datetime import datetime

import config

ALLOWED_TYPES = {"cleaning", "fault", "capacity_test", "maintenance", "other"}


def _conn():
    return sqlite3.connect(config.EVENTS_DB_PATH)


def init_events_table():
    """Create the events table if it doesn't exist (idempotent)."""
    conn = _conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plant_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type      TEXT NOT NULL,
                asset_id        TEXT,
                event_timestamp TEXT,
                value           REAL,
                notes           TEXT,
                source          TEXT DEFAULT 'manual',
                created_at      TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON plant_events(event_type);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_asset ON plant_events(asset_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON plant_events(event_timestamp);")
        conn.commit()
    finally:
        conn.close()


def log_event(event_type, asset_id=None, event_timestamp=None, value=None, notes=None, source="manual"):
    """Record one ground-truth event. Returns the stored row as a dict."""
    init_events_table()
    et = (event_type or "other").strip().lower()
    if et not in ALLOWED_TYPES:
        et = "other"
    ts = event_timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    created = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    val = None
    if value is not None and value != "":
        try:
            val = float(value)
        except (TypeError, ValueError):
            val = None
    conn = _conn()
    try:
        cur = conn.execute(
            "INSERT INTO plant_events (event_type, asset_id, event_timestamp, value, notes, source, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (et, asset_id, ts, val, notes, source, created),
        )
        conn.commit()
        event_id = cur.lastrowid
    finally:
        conn.close()
    return {
        "id": event_id, "event_type": et, "asset_id": asset_id,
        "event_timestamp": ts, "value": val, "notes": notes,
        "source": source, "created_at": created,
    }


def list_events(event_type=None, asset_id=None, start=None, end=None, limit=200):
    """Return recorded events, newest first, with optional filters."""
    init_events_table()
    query = ("SELECT id, event_type, asset_id, event_timestamp, value, notes, source, created_at "
             "FROM plant_events WHERE 1=1")
    params = []
    if event_type:
        query += " AND event_type = ?"
        params.append(event_type.strip().lower())
    if asset_id:
        query += " AND asset_id = ?"
        params.append(asset_id)
    if start:
        query += " AND event_timestamp >= ?"
        params.append(start)
    if end:
        query += " AND event_timestamp <= ?"
        params.append(end)
    query += " ORDER BY event_timestamp DESC, id DESC LIMIT ?"
    params.append(int(limit))
    conn = _conn()
    try:
        cur = conn.execute(query, params)
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()
    return rows


def summary():
    """Counts by event type — useful for dashboards / readiness checks."""
    init_events_table()
    conn = _conn()
    try:
        cur = conn.execute("SELECT event_type, COUNT(*) FROM plant_events GROUP BY event_type")
        by_type = {row[0]: row[1] for row in cur.fetchall()}
        total = sum(by_type.values())
    finally:
        conn.close()
    return {"total_events": total, "by_type": by_type}


def init_feedback_table():
    """Create the feedback table for capturing user ratings on LLM responses."""
    conn = _conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_feedback (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt          TEXT,
                response        TEXT,
                rating          INTEGER,
                label           TEXT,
                created_at      TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def log_feedback(prompt, response, rating, label=None):
    """Record user feedback on a response."""
    init_feedback_table()
    created = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = _conn()
    try:
        cur = conn.execute(
            "INSERT INTO user_feedback (prompt, response, rating, label, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (prompt, response, rating, label, created),
        )
        conn.commit()
        fb_id = cur.lastrowid
    finally:
        conn.close()
    return {"id": fb_id, "rating": rating, "label": label, "created_at": created}


def init_config_table():
    """Create and initialize the app configuration table."""
    conn = _conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_config (
                key     TEXT PRIMARY KEY,
                value   TEXT NOT NULL
            )
            """
        )
        defaults = {
            "PLANT_CAPACITY_KW": "8648.0",
            "LOSS_FACTOR": "0.85",
            "AVG_CURTAILED_KW": "4000.0",
            "HARDWARE_LOSS_SHARE": "0.25"
        }
        for k, v in defaults.items():
            conn.execute("INSERT OR IGNORE INTO app_config (key, value) VALUES (?, ?)", (k, v))
        conn.commit()
    finally:
        conn.close()


def get_config():
    """Get all configuration keys and values."""
    init_config_table()
    conn = _conn()
    try:
        cur = conn.execute("SELECT key, value FROM app_config")
        res = {row[0]: float(row[1]) for row in cur.fetchall()}
    finally:
        conn.close()
    return res


def update_config(config_dict):
    """Update configurations."""
    init_config_table()
    conn = _conn()
    try:
        for k, v in config_dict.items():
            conn.execute("INSERT OR REPLACE INTO app_config (key, value) VALUES (?, ?)", (k, str(v)))
        conn.commit()
    finally:
        conn.close()
    return get_config()


if __name__ == "__main__":
    init_events_table()
    init_feedback_table()
    init_config_table()
    print("Events DB initialized at", config.EVENTS_DB_PATH)
    print("Summary:", summary())
    print("Config:", get_config())
