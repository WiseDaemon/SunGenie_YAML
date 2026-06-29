import sqlite3
import json
import os
import sys
from datetime import datetime
import hashlib

# Ensure scratch directory is in python path
sys.path.append(os.path.dirname(__file__))
import ml_pipelines
import config
import events_store

DB_PATH = config.DB_PATH

# 1. SQL Query execution tool
def execute_sql_query(sql_query: str) -> str:
    """Executes a SQL SELECT query against the SunGenie telemetry database.
    
    Use this to look up specific raw telemetry columns, calculate averages, sums,
    or min/max over time, and filter by device_group, meterId, type, or timestamp.
    Only SELECT statements are allowed.
    
    Available Columns in the 'telemetry' table:
    - timestamp (TEXT, format 'YYYY-MM-DD HH:MM:SS')
    - type (TEXT, e.g. 'electricity_metering_data')
    - meterId (TEXT, e.g. 'JAMNAGAR_VIRTUAL_GATEWAY_B1INV1')
    - deviceUID (TEXT)
    - device_group (TEXT, e.g. 'WEATHER', 'INVERTER', 'METER', 'BESS', 'PCS', 'PQM', 'DCCON')
    - activePower (REAL)
    - energyToday (REAL)
    - energyTillDate (REAL)
    - apparentEnergyTillDate (REAL)
    - apparentEnergyToday (REAL)
    - inputPVPower (REAL)
    - inputCurrent (REAL)
    - inputVoltage (REAL)
    - inverterStatus (INTEGER)
    - bessSOC (REAL)
    - bessCurrent (REAL)
    - voltageRPhase (REAL)
    - currentRPhase (REAL)
    - totalActiveEnergyImport (REAL)
    - totalActiveEnergyExport (REAL)
    - netActiveEnergy (REAL)
    - windSpeed (REAL)
    - windDirection (REAL)
    - ambientTemperature (REAL)
    - humidity (REAL)
    - globalHorizontalIrradiance (REAL)
    - planeOfArraySensor01 (REAL)
    - moduleTemperatureSensor01 (REAL)
    - moduleTemperatureSensor03 (REAL)
    - rainfall (REAL)
    - scb_currents (TEXT, JSON string containing String Combiner Box currents)
    """
    # --- SEC-05: query hardening --------------------------------------------
    MAX_SQL_ROWS = 1000          # hard cap on rows scanned/returned
    cleaned = sql_query.strip().rstrip(";").strip()
    query_upper = cleaned.upper()

    if not query_upper.startswith("SELECT"):
        return json.dumps({"error": "Only SELECT queries are allowed for security."})

    # Reject stacked/multiple statements.
    if ";" in cleaned:
        return json.dumps({"error": "Only a single SELECT statement is allowed."})

    # Block statement types and functions that can mutate state, read the
    # filesystem, or attach other databases (defence in depth on top of SELECT-only).
    BLOCKED = ("ATTACH", "DETACH", "PRAGMA", "INSERT", "UPDATE", "DELETE",
               "DROP", "ALTER", "CREATE", "REPLACE", "VACUUM", "REINDEX",
               "LOAD_EXTENSION", "WRITEFILE", "READFILE", "EDITDIST3")
    if any(tok in query_upper for tok in BLOCKED):
        return json.dumps({"error": "Query contains a disallowed keyword."})

    # Wrap the user query so an absolute row cap always applies, regardless of
    # any LIMIT the caller did or did not include.
    capped_query = f"SELECT * FROM ({cleaned}) AS _sub LIMIT {MAX_SQL_ROWS}"

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.enable_load_extension(False)        # no extension loading
        conn.execute("PRAGMA query_only = ON;")  # read-only connection

        # Abort runaway queries: the handler fires every 100k VM instructions
        # and aborts once the budget is exhausted (~ a few seconds of CPU).
        budget = {"n": 0}
        def _watchdog():
            budget["n"] += 1
            return 1 if budget["n"] > 2000 else 0   # ~200M instructions
        conn.set_progress_handler(_watchdog, 100000)

        cursor = conn.cursor()
        cursor.execute(capped_query)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()

        result = [dict(zip(columns, row)) for row in rows]
        # Truncate the *serialised* payload to avoid token overflow downstream.
        if len(result) > 50:
            return json.dumps(result[:50]) + "\n... (truncated, showing 50 of " + str(len(result)) + " rows; max " + str(MAX_SQL_ROWS) + ")"
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        if conn is not None:
            conn.close()

# 2. PR Gap analysis tool
def get_pr_gap_analysis(date_str: str = None) -> str:
    """Calculates the Expected vs. Actual generation and decomposes the PR Gap (attributing it to Soiling, Shading, etc.).
    
    Args:
        date_str: Optional date string in 'YYYY-MM-DD' format (e.g. '2026-06-08').
    """
    try:
        res = ml_pipelines.get_expected_vs_actual_generation(date_str)
        return json.dumps(res)
    except Exception as e:
        return json.dumps({"error": str(e)})

# 3. BESS health analysis tool
def get_bess_health_status(bess_id: str = "JAMNAGAR_VIRTUAL_GATEWAY_B1BCT1") -> str:
    """Calculates BESS Coulombic Efficiency, State of Health (SoH) percentage, and total operating cycles.
    
    Args:
        bess_id: The BESS meterId (e.g. 'JAMNAGAR_VIRTUAL_GATEWAY_B1BCT1').
    """
    try:
        res = ml_pipelines.get_bess_health(bess_id)
        return json.dumps(res)
    except Exception as e:
        return json.dumps({"error": str(e)})

# 4. Soiling rate calibration tool
def get_soiling_calibration() -> str:
    """Detects sudden PR jumps (panel cleaning events) and calculates average daily soiling rate."""
    try:
        res = ml_pipelines.calibrate_soiling_rate()
        return json.dumps(res)
    except Exception as e:
        return json.dumps({"error": str(e)})

# 5. System Uptime & ENS Tool
def get_system_uptime_ens() -> str:
    """Calculates Plant Uptime % and Energy Not Supplied (ENS) due to faults or curtailment."""
    try:
        res = ml_pipelines.get_system_uptime_ens()
        return json.dumps(res)
    except Exception as e:
        return json.dumps({"error": str(e)})

# 6. BESS RTE Tool
def get_bess_rte(bess_id: str = "JAMNAGAR_VIRTUAL_GATEWAY_B1BCT1") -> str:
    """Calculates BESS Round Trip Efficiency (RTE) to determine battery degradation."""
    try:
        res = ml_pipelines.get_bess_rte(bess_id)
        return json.dumps(res)
    except Exception as e:
        return json.dumps({"error": str(e)})

# 7. Tracker Misalignment Tool
def get_tracker_misalignment() -> str:
    """Estimates tracker misalignment losses and identifies affected trackers."""
    try:
        res = ml_pipelines.get_tracker_misalignment()
        return json.dumps(res)
    except Exception as e:
        return json.dumps({"error": str(e)})

# 5. Inverter string current outlier detection tool
def get_scb_outliers(inverter_id: str = "JAMNAGAR_VIRTUAL_GATEWAY_B1INV1", timestamp_str: str = None) -> str:
    """Analyzes string combiner box currents to isolate localized string-level failures or shading.
    
    Args:
        inverter_id: Inverter ID (e.g. 'JAMNAGAR_VIRTUAL_GATEWAY_B1INV1').
        timestamp_str: Optional timestamp (format 'YYYY-MM-DD HH:MM:SS'). Defaults to latest active reading.
    """
    try:
        res = ml_pipelines.detect_scb_outliers(inverter_id, timestamp_str)
        return json.dumps(res)
    except Exception as e:
        return json.dumps({"error": str(e)})

# 6. Actionable Task JSON payload generation tool
def generate_actionable_task_payload(title: str, severity: str, asset_id: str, description: str) -> str:
    """Generates a structured JSON ticket payload that the operator can copy-paste into the SunGenie Actionable Tasks board.
    
    Args:
        title: Short title of the task/issue.
        severity: Severity level ('Critical', 'Warning', 'Info').
        asset_id: Unique identifier of the asset (e.g. 'B1INV1').
        description: Detailed explanation of the fault and recommended action.
    """
    payload = {
        "ticketId": f"TK-{hashlib.md5(title.encode()).hexdigest()[:6].upper()}",
        "title": title,
        "severity": severity,
        "assetId": asset_id,
        "description": description,
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "Open",
        "aging_days": 0,
        "source": "AI_Agent_Diagnostic"
    }
    return json.dumps(payload, indent=2)

# 7. Ground-truth event logging tools (label capture — foundation for supervised ML)
def log_plant_event(event_type: str, asset_id: str = None, event_timestamp: str = None, value: float = None, notes: str = None) -> str:
    """Records a ground-truth operational event (label) for future ML training.

    Use this whenever the operator reports that something physically happened that the
    telemetry does not capture — a panel cleaning, a confirmed fault, a BESS capacity
    test, or maintenance. These labels are the prerequisite for predictive models.

    Args:
        event_type: One of 'cleaning', 'fault', 'capacity_test', 'maintenance', 'other'.
        asset_id: Affected asset id (e.g. 'JAMNAGAR_VIRTUAL_GATEWAY_B1INV1'), optional.
        event_timestamp: When it happened ('YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'); defaults to now.
        value: Optional numeric (e.g. measured capacity %, fault code).
        notes: Free-text detail.
    """
    try:
        rec = events_store.log_event(event_type, asset_id, event_timestamp, value, notes, source="AI_Agent")
        return json.dumps({"status": "logged", "event": rec})
    except Exception as e:
        return json.dumps({"error": str(e)})

def get_plant_events(event_type: str = None, asset_id: str = None) -> str:
    """Lists previously recorded ground-truth events (cleanings, faults, capacity tests).

    Args:
        event_type: Optional filter ('cleaning', 'fault', 'capacity_test', 'maintenance', 'other').
        asset_id: Optional asset id filter (e.g. 'JAMNAGAR_VIRTUAL_GATEWAY_B1BCT1').
    """
    try:
        rows = events_store.list_events(event_type=event_type, asset_id=asset_id)
        return json.dumps(rows)
    except Exception as e:
        return json.dumps({"error": str(e)})

def get_agent_config():
    try:
        import sys
        if sys.platform == "win32":
            raise RuntimeError("Windows requires localharness shim")
        from google.antigravity import LocalAgentConfig
    except (ImportError, RuntimeError):
        from google_antigravity_shim import LocalAgentConfig
    
    # Configure the agent with our tools and system instructions
    config = LocalAgentConfig(
        tools=[
            execute_sql_query,
            get_pr_gap_analysis,
            get_bess_health_status,
            get_soiling_calibration,
            get_system_uptime_ens,
            get_bess_rte,
            get_tracker_misalignment,
            get_scb_outliers,
            generate_actionable_task_payload,
            log_plant_event,
            get_plant_events
        ],
        system_instructions=(
            "You are the SunGenie AI Assistant (eAnalytiX Platform), an expert AI/ML solar engineering assistant.\n\n"
            "Your goal is to help operators monitor the Jamnagar Central Solar Plant using raw telemetry data.\n"
            "You have tools to perform PR gap analysis, BESS health diagnostics, BESS round trip efficiency (RTE), "
            "tracker misalignment analysis, system uptime (ENS), soiling rate calibration, "
            "string current outlier analysis, and execute custom SQL SELECT queries.\n\n"
            "GUIDELINES:\n"
            "1. When answering queries about performance or drops, prioritize using the high-level tools first "
            "(get_pr_gap_analysis, get_soiling_calibration, get_scb_outliers, get_bess_health_status, "
            "get_system_uptime_ens, get_bess_rte, get_tracker_misalignment).\n"
            "2. Use execute_sql_query for custom aggregations, specific timestamp lookups, or weather queries.\n"
            "3. If you find underperforming assets or faults (e.g. a string current outlier or battery SOC drift), "
            "use generate_actionable_task_payload to generate a JSON work order and present it to the operator "
            "so they can copy-paste it into the Actionable Tasks board.\n"
            "4. If the operator reports a real-world event the telemetry can't see (a panel cleaning, a confirmed "
            "fault, a BESS capacity test, or maintenance), call log_plant_event to record it; use get_plant_events "
            "to recall history. These labels train future predictive models.\n"
            "5. CRITICAL: NEVER output your internal reasoning, goals, or planning steps (e.g., 'Agent Thoughts', 'User says...', 'Rule 1:'). Start your response directly with the final message to the user.\n"
            "6. When a user says 'hi', 'hello', or makes small talk, reply warmly and conversationally in 1-2 sentences. Do NOT list out your capabilities unless explicitly asked, and do NOT be robotic."
        )
    )
    return config

if __name__ == "__main__":
    # Test agent import and config
    try:
        import sys
        if sys.platform == "win32":
            raise RuntimeError("Windows requires localharness shim")
        from google.antigravity import Agent
        print("Google Antigravity SDK imported successfully!")
    except (ImportError, RuntimeError):
        from google_antigravity_shim import Agent
        print("Using Google Antigravity Windows Compatibility Shim...")
    config = get_agent_config()
    print("Agent configuration built successfully!")
