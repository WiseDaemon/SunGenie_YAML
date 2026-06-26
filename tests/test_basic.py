"""SunGenie regression tests (ENG-03).

Focus on the spots that have broken before:
- relative-date resolution (BUG-02)
- device-group mapping (QUA-01)
- deterministic ticket IDs (BUG-04)
- the server actually imports (BLOCK-01 regression guard)
- SQL hardening rejects non-SELECT / stacked statements (SEC-05)
- PR-gap attribution sums to the gap (DB-gated; skipped if no DB)
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- BUG-02: relative dates -------------------------------------------------
def test_today_and_yesterday_differ():
    from google_antigravity_shim import resolve_timeframe
    today_start, _ = resolve_timeframe("show me today")
    yday_start, _ = resolve_timeframe("show me yesterday")
    assert today_start is not None and yday_start is not None
    assert today_start[:10] != yday_start[:10]


# --- QUA-01: device-group mapping ------------------------------------------
@pytest.mark.parametrize("meter,expected", [
    ("JAMNAGAR_VIRTUAL_GATEWAY_B1INV1", "INVERTER"),
    ("JAMNAGAR_VIRTUAL_GATEWAY_B2MFM12", "METER"),
    ("JAMNAGAR_VIRTUAL_GATEWAY_B1BCT1", "BESS"),
    ("JAMNAGAR_VIRTUAL_GATEWAY_B1PCS1", "PCS"),
    ("JAMNAGAR_VIRTUAL_GATEWAY_PPCWMS1", "WEATHER"),
    ("JAMNAGAR_VIRTUAL_GATEWAY_B2DCCON1", "DCCON"),
    ("", "OTHER"),
])
def test_determine_device_group(meter, expected):
    from utils import determine_device_group
    assert determine_device_group(meter) == expected


# --- BUG-04: deterministic ticket IDs --------------------------------------
def test_ticket_id_is_deterministic():
    from agent_setup import generate_actionable_task_payload
    import json
    a = json.loads(generate_actionable_task_payload("Same Title", "Warning", "B1INV1", "desc"))
    b = json.loads(generate_actionable_task_payload("Same Title", "Warning", "B1INV1", "desc"))
    assert a["ticketId"] == b["ticketId"]


# --- BLOCK-01: the server must import without raising ----------------------
def test_app_server_imports():
    import importlib
    mod = importlib.import_module("app_server")
    assert hasattr(mod, "app")


# --- SEC-05: SQL guard rejects unsafe queries ------------------------------
def test_sql_rejects_non_select():
    from agent_setup import execute_sql_query
    import json
    out = json.loads(execute_sql_query("DROP TABLE telemetry"))
    assert "error" in out


def test_sql_rejects_stacked_statements():
    from agent_setup import execute_sql_query
    import json
    out = json.loads(execute_sql_query("SELECT 1; SELECT 2"))
    assert "error" in out


# --- LOG-01b: PR-gap attribution sums to the gap (needs the DB) ------------
def _db_available():
    import config
    if not os.path.exists(config.DB_PATH):
        return False
    try:
        import sqlite3
        c = sqlite3.connect(config.DB_PATH)
        n = c.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0]
        c.close()
        return n > 0
    except Exception:
        return False


@pytest.mark.skipif(not _db_available(), reason="telemetry DB not present")
def test_pr_gap_attribution_sums_to_gap():
    import ml_pipelines
    res = ml_pipelines.get_expected_vs_actual_generation()
    gap = res["gap_kwh"]
    if gap <= 0:
        pytest.skip("no positive gap in this dataset")
    total = sum(res["attribution"].values())
    assert abs(total - gap) <= max(1.0, 0.02 * gap)  # within 2% (rounding)


@pytest.mark.skipif(not _db_available(), reason="telemetry DB not present")
def test_bess_health_reports_estimated_flag():
    import ml_pipelines
    res = ml_pipelines.get_bess_health()
    # Either a real estimate (with the flag) or an explicit insufficient-data signal.
    assert res.get("estimated") is True or res.get("status") == "Insufficient Data"
