import sqlite3
import json
import os
import sys
from datetime import datetime

# Ensure scratch directory is in python path
sys.path.append(os.path.dirname(__file__))
import ml_pipelines

DB_PATH = r"C:\Users\saxen\.gemini\antigravity\brain\0f95910d-307c-40cb-b421-02dc23fbd684\scratch\sungenie_telemetry.db"

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
    query_upper = sql_query.strip().upper()
    if not query_upper.startswith("SELECT"):
        return json.dumps({"error": "Only SELECT queries are allowed for security."})
        
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        
        result = [dict(zip(columns, row)) for row in rows]
        # Return truncated list if too long to prevent token overflow
        if len(result) > 50:
            return json.dumps(result[:50]) + "\n... (truncated, total " + str(len(result)) + " rows)"
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

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
        "ticketId": f"TK-{hash(title) % 1000000:06d}",
        "title": title,
        "severity": severity,
        "assetId": asset_id,
        "description": description,
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ") if 'datetime' in globals() else "2026-06-24T17:52:00Z",
        "status": "Open",
        "aging_days": 0,
        "source": "AI_Agent_Diagnostic"
    }
    return json.dumps(payload, indent=2)

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
            get_scb_outliers,
            generate_actionable_task_payload
        ],
        system_instructions=(
            "You are the SunGenie AI Assistant (eAnalytiX Platform), an expert AI/ML solar engineering assistant.\n\n"
            "Your goal is to help operators monitor the Jamnagar Central Solar Plant using raw telemetry data.\n"
            "You have tools to perform PR gap analysis, BESS health diagnostics, soiling rate calibration, "
            "string current outlier analysis, and execute custom SQL SELECT queries.\n\n"
            "GUIDELINES:\n"
            "1. When answering queries about performance or drops, prioritize using the high-level tools first "
            "(get_pr_gap_analysis, get_soiling_calibration, get_scb_outliers, get_bess_health_status).\n"
            "2. Use execute_sql_query for custom aggregations, specific timestamp lookups, or weather queries.\n"
            "3. If you find underperforming assets or faults (e.g. a string current outlier or battery SOC drift), "
            "use generate_actionable_task_payload to generate a JSON work order and present it to the operator "
            "so they can copy-paste it into the Actionable Tasks board.\n"
            "4. Keep your responses concise, technical, and data-backed."
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
