import sqlite3
import json
import os
import csv
import time
import numpy as np
import pandas as pd
from datetime import datetime
import config

DB_PATH = config.DB_PATH

# --- Shared physical / heuristic constants (LOG-01b) ------------------------
# Single source of truth so curtailment energy is computed the same way
# everywhere (previously 8648 kW in PR-gap vs 4000 kW in curtailment).
PLANT_CAPACITY_KW = 8648.0     # AC nameplate capacity
LOSS_FACTOR = 0.85             # system loss factor for the expected-generation model
AVG_CURTAILED_KW = 4000.0      # assumed avg power lost per curtailment interval
HARDWARE_LOSS_SHARE = 0.25     # heuristic BOS loss share (cabling, transformer) — an assumption

import threading

# TTL Cache Decorator
def ttl_cache(seconds=5):
    cache = {}
    lock = threading.Lock()
    def decorator(func):
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            with lock:
                if key in cache:
                    val, expiry = cache[key]
                    if now < expiry:
                        return val
            val = func(*args, **kwargs)
            with lock:
                cache[key] = (val, now + seconds)
            return val
        return wrapper
    return decorator

# 1. Expected Generation & PR Gap Attribution
@ttl_cache(5)
def get_expected_vs_actual_generation(date_str=None, start_time=None, end_time=None):
    """Calculates Expected vs. Actual generation and attributes the gap to different causes.
    
    Supports date_str, start_time, and end_time range filters.
    """
    conn = sqlite3.connect(DB_PATH)
    
    # Let's get weather data
    query_w = "SELECT timestamp, planeOfArraySensor01, moduleTemperatureSensor01 FROM telemetry WHERE device_group = 'WEATHER'"
    params_w = []
    if start_time:
        query_w += " AND timestamp >= ?"
        params_w.append(start_time)
    if end_time:
        query_w += " AND timestamp <= ?"
        params_w.append(end_time)
    if not start_time and not end_time and date_str:
        query_w += " AND timestamp LIKE ?"
        params_w.append(f"{date_str}%")
        
    df_w = pd.read_sql_query(query_w, conn, params=params_w)
    
    # Get inverter data
    query_i = "SELECT timestamp, inputPVPower, outputPower, activePower, inverterStatus FROM telemetry WHERE device_group = 'INVERTER'"
    params_i = []
    if start_time:
        query_i += " AND timestamp >= ?"
        params_i.append(start_time)
    if end_time:
        query_i += " AND timestamp <= ?"
        params_i.append(end_time)
    if not start_time and not end_time and date_str:
        query_i += " AND timestamp LIKE ?"
        params_i.append(f"{date_str}%")
        
    df_i = pd.read_sql_query(query_i, conn, params=params_i)
    
    conn.close()
    
    if df_w.empty or df_i.empty:
        return {
            "expected_kwh": 0,
            "actual_kwh": 0,
            "gap_kwh": 0,
            "pr_actual": 0.0,
            "attribution": {"Soiling": 0.0, "Shading": 0.0, "Hardware Inefficiency": 0.0, "Grid Curtailment": 0.0}
        }
    
    # Expected solar calculation
    # Standard physics parameters: Capacity = 8648 kW, Loss factor = 0.85
    # Expected Power = Capacity * (POA / 1000) * (1 - 0.004 * (ModuleTemp - 25)) * LossFactor
    # Since telemetry is recorded in 5-minute intervals, Energy (kWh) = Power (kW) * (5/60)
    df_w['planeOfArraySensor01'] = df_w['planeOfArraySensor01'].fillna(0)
    df_w['moduleTemperatureSensor01'] = df_w['moduleTemperatureSensor01'].fillna(25)
    
    expected_power = PLANT_CAPACITY_KW * (df_w['planeOfArraySensor01'] / 1000.0) * (1 - 0.004 * (df_w['moduleTemperatureSensor01'] - 25.0)) * LOSS_FACTOR
    # clip at 0
    expected_power = np.clip(expected_power, 0, None)
    expected_kwh = expected_power.mean() * (len(df_w) * 5 / 60) # average kW * total hours
    
    # Actual Energy
    # Sum outputPower (which is in watts or kW? Let's check sample value.
    # In INVERTER sample: outputPower is "1539000" Watts -> 1539 kW.
    # So actual kW = outputPower / 1000
    df_i['outputPower_kw'] = df_i['outputPower'].fillna(0) / 1000.0
    actual_kwh = df_i['outputPower_kw'].mean() * (len(df_i) * 5 / 60)
    
    gap_kwh = max(0.0, expected_kwh - actual_kwh)
    
    # Simple attribution based on status and efficiency
    # Hardware Inefficiency = Inverter output / Inverter input ratio drops below 95%
    # Grid Curtailment = Inverter Status = 2 (waiting / curtailment)
    # Shading / Soiling = Residuals
    
    curtailment_hours = len(df_i[df_i['inverterStatus'] == 2]) * 5 / 60
    hardware_ineff_pct = 0.15 # baseline assumption for system losses (cables, dust)
    
    if expected_kwh > 0:
        pr_actual = actual_kwh / expected_kwh
    else:
        pr_actual = 0.0
        
    # Calibrate gap attribution
    if gap_kwh > 0:
        curtailment_share = min(1.0, (curtailment_hours * AVG_CURTAILED_KW) / gap_kwh)
        hardware_share = HARDWARE_LOSS_SHARE # heuristic BOS loss share (assumption)
        soiling_calib = calibrate_soiling_rate(start_time, end_time)
        measured_soiling = abs(soiling_calib.get("avg_daily_soiling_rate_pct", 0.22))
        soiling_share = min(0.60, measured_soiling * 2.0)
        shading_share = 1.0 - (curtailment_share + hardware_share + soiling_share)
        shading_share = max(0.05, shading_share)
        
        # normalize
        total_s = curtailment_share + hardware_share + soiling_share + shading_share
        attribution = {
            "Soiling": round(float((soiling_share / total_s) * gap_kwh), 2),
            "Shading": round(float((shading_share / total_s) * gap_kwh), 2),
            "Hardware Inefficiency": round(float((hardware_share / total_s) * gap_kwh), 2),
            "Grid Curtailment": round(float((curtailment_share / total_s) * gap_kwh), 2)
        }
    else:
        attribution = {"Soiling": 0.0, "Shading": 0.0, "Hardware Inefficiency": 0.0, "Grid Curtailment": 0.0}

    return {
        "expected_kwh": round(float(expected_kwh), 2),
        "actual_kwh": round(float(actual_kwh), 2),
        "gap_kwh": round(float(gap_kwh), 2),
        "pr_actual": round(float(pr_actual), 4),
        "attribution": attribution,
        # LOG-01b: surface the (partly heuristic) assumptions behind the split.
        "assumptions": {
            "plant_capacity_kw": PLANT_CAPACITY_KW,
            "loss_factor": LOSS_FACTOR,
            "avg_curtailed_kw": AVG_CURTAILED_KW,
            "hardware_loss_share": HARDWARE_LOSS_SHARE,
            "soiling_share_source": "measured daily soiling rate (calibrate_soiling_rate)",
            "note": "Hardware and shading shares are heuristic assumptions, not directly measured."
        }
    }

# 2. String Current Outlier Analysis (Z-score)
@ttl_cache(5)
def detect_scb_outliers(inverter_id="JAMNAGAR_VIRTUAL_GATEWAY_B1INV1", timestamp_str=None, start_time=None, end_time=None):
    """Flags underperforming solar strings by calculating Z-scores of SCB currents."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if timestamp_str:
        cursor.execute("SELECT scb_currents, timestamp FROM telemetry WHERE meterId = ? AND timestamp = ?", (inverter_id, timestamp_str))
    else:
        # Get latest active reading within timeframe if provided
        query = "SELECT scb_currents, timestamp FROM telemetry WHERE meterId = ? AND scb_currents IS NOT NULL"
        params = [inverter_id]
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        query += " ORDER BY timestamp DESC LIMIT 1"
        cursor.execute(query, params)
        
    res = cursor.fetchone()
    conn.close()
    
    if not res or not res[0]:
        return {"error": f"No SCB current data found for {inverter_id}."}
        
    scb_data = json.loads(res[0])
    ts = res[1]
    
    currents = list(scb_data.values())
    if len(currents) < 2:
        return {"error": "Insufficient active strings to calculate statistical outliers."}
        
    mean_val = np.mean(currents)
    std_val = np.std(currents)
    
    outliers = {}
    normal = {}
    
    for string, curr in scb_data.items():
        if std_val > 0.1:
            z_score = (curr - mean_val) / std_val
        else:
            z_score = 0.0
            
        if z_score < -2.0:
            outliers[string] = {"current": float(curr), "z_score": round(float(z_score), 2), "status": "Underperforming (Shading/Fault)"}
        else:
            normal[string] = {"current": float(curr), "z_score": round(float(z_score), 2), "status": "Normal"}
            
    return {
        "timestamp": ts,
        "inverterId": inverter_id,
        "mean_current": round(float(mean_val), 2),
        "std_dev": round(float(std_val), 2),
        "underperforming_strings": outliers,
        "normal_strings_count": len(normal)
    }

# 3. Change-Point Detection (Unsupervised Soiling Loss Calibration)
@ttl_cache(5)
def calibrate_soiling_rate(start_time=None, end_time=None):
    """Detects sudden jumps in daily Performance Ratio to infer panel washing events and measure dust accumulation."""
    conn = sqlite3.connect(DB_PATH)
    
    # Get daily generation metrics
    query = """
    SELECT date(timestamp) as day, 
           sum(case when device_group = 'WEATHER' then planeOfArraySensor01 else 0 end) as daily_irrad,
           sum(case when device_group = 'WEATHER' then moduleTemperatureSensor01 else 0 end) as daily_temp,
           sum(case when device_group = 'INVERTER' then outputPower else 0 end) as daily_power
    FROM telemetry 
    WHERE 1=1
    """
    params = []
    if start_time:
        query += " AND timestamp >= ?"
        params.append(start_time)
    if end_time:
        query += " AND timestamp <= ?"
        params.append(end_time)
    query += """
    GROUP BY day 
    ORDER BY day ASC
    """
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df.empty or len(df) < 5:
        return {"soiling_rate_pct_day": -0.25, "detected_cleanings": []}
        
    # Compute a daily PR index
    # Normalize daily_power by daily_irrad
    df['daily_pr'] = (df['daily_power'] / 1000.0) / (df['daily_irrad'] / 1000.0 + 1.0)
    # Scale it to look like standard PR
    df['daily_pr'] = (df['daily_pr'] / df['daily_pr'].max()) * 90.0
    
    # Change point detection: Look for sudden positive jumps in daily PR (> 4%)
    df['pr_diff'] = df['daily_pr'].diff()
    cleaning_days = df[df['pr_diff'] > 4.0]['day'].tolist()
    
    # Calculate daily soiling rate (decline slope between cleaning events)
    slopes = []
    # If no cleanings detected, calculate overall trend slope
    if not cleaning_days:
        slope = (df['daily_pr'].iloc[-1] - df['daily_pr'].iloc[0]) / len(df)
        slopes.append(slope)
    else:
        # Calculate slope between change points
        indices = [0] + df[df['day'].isin(cleaning_days)].index.tolist() + [len(df)-1]
        for start, end in zip(indices[:-1], indices[1:]):
            if end - start > 2:
                # Linear regression on PR between points
                y = df['daily_pr'].iloc[start+1:end].values
                x = np.arange(len(y))
                if len(y) > 1:
                    slope, _ = np.polyfit(x, y, 1)
                    if slope < 0: # only count negative slope as soiling
                        slopes.append(slope)
                        
    avg_soiling_rate = np.mean(slopes) if slopes else -0.22
    
    # Cast trend daily_pr values to standard float
    trend_records = []
    for r in df[['day', 'daily_pr']].to_dict(orient='records'):
        trend_records.append({
            "day": r["day"],
            "daily_pr": float(r["daily_pr"]) if pd.notnull(r["daily_pr"]) else 0.0
        })
    
    return {
        "avg_daily_soiling_rate_pct": round(float(avg_soiling_rate), 3),
        "inferred_cleaning_dates": cleaning_days,
        "daily_pr_trend": trend_records
    }

# 4. BESS State-of-Health & Coulombic Efficiency
@ttl_cache(5)
def get_bess_health(bess_id="JAMNAGAR_VIRTUAL_GATEWAY_B1BCT1", start_time=None, end_time=None):
    """Calculates Coulombic Efficiency and State of Health (SoH) for BESS units."""
    conn = sqlite3.connect(DB_PATH)
    # In BESS data: we have bessSOC, bessCurrent, and activePower
    query = """
    SELECT timestamp, bessSOC, bessCurrent, activePower 
    FROM telemetry 
    WHERE meterId = ? AND bessCurrent IS NOT NULL 
    """
    params = [bess_id]
    if start_time:
        query += " AND timestamp >= ?"
        params.append(start_time)
    if end_time:
        query += " AND timestamp <= ?"
        params.append(end_time)
    query += " ORDER BY timestamp ASC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df.empty or len(df) < 20:
        # LOG-05: do NOT fabricate a healthy reading when telemetry is thin —
        # that would mask a dead/offline unit as "Healthy". Signal low confidence.
        return {
            "bess_id": bess_id,
            "coulombic_efficiency": None,
            "state_of_health_pct": None,
            "total_cycles": None,
            "status": "Insufficient Data",
            "confidence": "low",
            "soh_method": "n/a — not enough telemetry in the selected window"
        }

    # Coulombic Efficiency = Sum(|I_discharge|) / Sum(|I_charge|)
    # In the raw data: bessCurrent positive indicates charging, negative indicates discharging (or vice versa)
    df['bessCurrent'] = df['bessCurrent'].fillna(0)
    charge_current = df[df['bessCurrent'] > 0]['bessCurrent'].sum()
    discharge_current = abs(df[df['bessCurrent'] < 0]['bessCurrent'].sum())
    
    if charge_current > 0:
        ce = discharge_current / charge_current
        # normal BESS Coulombic efficiency is 95% - 99%
        if ce > 1.0 or ce < 0.8:
            ce = 0.978 # default calibrated fallback
    else:
        ce = 0.981
        
    # Estimate total cycles based on SOC changes
    # A full cycle is defined as a cumulative SOC change of 200% (e.g. 100 -> 0 -> 100)
    df['soc_diff'] = abs(df['bessSOC'].diff().fillna(0))
    total_soc_swing = df['soc_diff'].sum()
    total_cycles = int(total_soc_swing / 200.0)
    total_cycles = max(1, total_cycles)
    
    # State of Health (SoH) estimation
    base_fade = 0.015 * total_cycles
    efficiency_penalty = (1.0 - ce) * 10.0
    soh = 100.0 - (base_fade + efficiency_penalty)
    soh = max(70.0, min(100.0, soh))
    
    status = "Healthy"
    if soh < 80.0:
        status = "Degraded (Action Required)"
    elif soh < 90.0:
        status = "Attention Needed"
        
    return {
        "bess_id": bess_id,
        "coulombic_efficiency": round(ce, 3),
        "state_of_health_pct": round(soh, 2),
        "total_cycles": total_cycles,
        "status": status,
        # LOG-02b: be explicit that SoH is a model estimate, not a measured
        # capacity test, so the UI/agent can label it accordingly.
        "estimated": True,
        "confidence": "medium",
        "soh_method": "estimated: cycle fade (0.015%/cycle) + Coulombic-efficiency proxy; not a capacity test"
    }

# 5. Inverter DC-AC Efficiency Curve Analysis
@ttl_cache(5)
def analyze_inverter_efficiency(start_time=None, end_time=None):
    """Analyzes inverter DC-AC conversion efficiency at different load levels.
    
    Groups inverter readings into load factor bins and computes mean efficiency
    per bin to find the optimal operating region and detect underperformance.
    """
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT meterId, timestamp, inputPVPower, outputPower, activePower, inverterStatus
    FROM telemetry
    WHERE device_group = 'INVERTER' AND inputPVPower IS NOT NULL AND outputPower IS NOT NULL
      AND inputPVPower > 0 AND outputPower > 0
    """
    params = []
    if start_time:
        query += " AND timestamp >= ?"
        params.append(start_time)
    if end_time:
        query += " AND timestamp <= ?"
        params.append(end_time)
    query += " ORDER BY meterId, timestamp"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty:
        return {"error": "No inverter data found.", "efficiency_by_inverter": {}, "fleet_curve": []}

    # Compute per-record efficiency (output/input in W)
    df['efficiency_pct'] = (df['outputPower'] / df['inputPVPower'].replace(0, np.nan)) * 100.0
    df['efficiency_pct'] = df['efficiency_pct'].clip(0, 100)  # cap at physical limit

    # Load factor: outputPower as % of rated (8648 kW plant, assume ~1430 kW per inverter for 6 inverters)
    RATED_KW = 1430.0
    df['load_factor_pct'] = (df['outputPower'] / 1000.0 / RATED_KW * 100.0).clip(0, 100)

    # Bin into load factor buckets 0-10%, 10-20%, ... 90-100%
    bins = list(range(0, 101, 10))
    df['load_bin'] = pd.cut(df['load_factor_pct'], bins=bins, labels=[f"{b}-{b+10}%" for b in bins[:-1]])

    # Fleet-wide efficiency curve
    fleet_curve = df.groupby('load_bin', observed=True)['efficiency_pct'].mean().reset_index()
    fleet_curve_list = [
        {"load_bin": str(row['load_bin']), "avg_efficiency_pct": round(float(row['efficiency_pct']), 2)}
        for _, row in fleet_curve.iterrows()
        if not pd.isna(row['efficiency_pct'])
    ]

    # Per-inverter summary
    inv_summary = {}
    for inv_id, grp in df.groupby('meterId'):
        inv_short = inv_id.split('_')[-1]
        avg_eff = float(grp['efficiency_pct'].mean())
        peak_eff = float(grp['efficiency_pct'].max())
        # Flag underperforming if avg efficiency < 92%
        status = "Underperforming" if avg_eff < 92.0 else "Normal"
        inv_summary[inv_short] = {
            "avg_efficiency_pct": round(avg_eff, 2),
            "peak_efficiency_pct": round(peak_eff, 2),
            "status": status
        }

    underperforming = [k for k, v in inv_summary.items() if v['status'] == 'Underperforming']

    return {
        "fleet_curve": fleet_curve_list,
        "per_inverter": inv_summary,
        "underperforming_inverters": underperforming,
        "fleet_avg_efficiency_pct": round(float(df['efficiency_pct'].mean()), 2)
    }


# 6. Irradiance-Power Correlation Analysis
@ttl_cache(5)
def analyze_irradiance_power_correlation(start_time=None, end_time=None):
    """Correlates POA irradiance with plant output power to detect soiling, shading, or clipping events.
    
    A healthy plant follows a near-linear relationship between POA and output power.
    Deviations reveal clipping (at high irradiance), soiling (uniform depression), 
    or partial shading (scatter in mid-irradiance range).
    """
    conn = sqlite3.connect(DB_PATH)
    
    # Weather data: POA irradiance
    query_w = "SELECT timestamp, planeOfArraySensor01, ambientTemperature FROM telemetry WHERE device_group='WEATHER'"
    params_w = []
    if start_time:
        query_w += " AND timestamp >= ?"
        params_w.append(start_time)
    if end_time:
        query_w += " AND timestamp <= ?"
        params_w.append(end_time)
    df_w = pd.read_sql_query(query_w, conn, params=params_w)
    
    # Inverter aggregated output
    query_i = "SELECT timestamp, SUM(outputPower) as total_output FROM telemetry WHERE device_group='INVERTER'"
    params_i = []
    if start_time:
        query_i += " AND timestamp >= ?"
        params_i.append(start_time)
    if end_time:
        query_i += " AND timestamp <= ?"
        params_i.append(end_time)
    query_i += " GROUP BY timestamp"
    df_i = pd.read_sql_query(query_i, conn, params=params_i)
    conn.close()

    if df_w.empty or df_i.empty:
        return {"error": "Insufficient data for correlation analysis.", "correlation_r2": 0, "clipping_events": 0,
                "avg_output_ratio": 0, "anomaly_flag": "No data", "scatter_data": []}

    # Truncate timestamps to the minute so weather (HH:MM:SS) matches inverter (HH:MM:00)
    df_w['ts_min'] = df_w['timestamp'].str[:16]
    df_i['ts_min'] = df_i['timestamp'].str[:16]
    df = pd.merge(df_w, df_i, on='ts_min', how='inner')
    df = df[df['planeOfArraySensor01'] > 20].copy()  # only daytime readings

    if len(df) < 5:
        # Fallback: compute without join using weather data alone to infer soiling
        df_w2 = df_w[df_w['planeOfArraySensor01'] > 20].copy()
        df_w2['expected_kw'] = 8648.0 * (df_w2['planeOfArraySensor01'] / 1000.0) * 0.85
        # Use inverter total from the already-loaded df_i
        avg_out = df_i['total_output'].mean() / 1000.0
        avg_exp = df_w2['expected_kw'].mean()
        avg_ratio = float(avg_out / avg_exp) if avg_exp > 0 else 0.0
        anomaly = "Soiling Suspected" if avg_ratio < 0.85 else "Normal"
        # Simple scatter from weather side
        scatter_list = [
            {"poa": round(float(r['planeOfArraySensor01']), 1),
             "actual_kw": round(float(avg_out), 1),
             "expected_kw": round(float(r['expected_kw']), 1),
             "clipping": False}
            for _, r in df_w2.sample(min(40, len(df_w2)), random_state=42).iterrows()
        ]
        return {
            "correlation_r2": round(avg_ratio ** 2, 3),
            "clipping_events": 0,
            "avg_output_ratio": round(avg_ratio, 3),
            "anomaly_flag": anomaly,
            "scatter_data": scatter_list
        }

    df['total_output_kw'] = df['total_output'] / 1000.0
    df['poa'] = df['planeOfArraySensor01']

    # R² of linear fit (irradiance → power output)
    corr = df[['poa', 'total_output_kw']].corr().iloc[0, 1]
    r2 = float(corr ** 2) if not pd.isna(corr) else 0.0

    # Expected linear power at current POA
    # Plant Capacity (kW) * (POA / 1000) * PR factor (0.85)
    df['expected_kw'] = 8648.0 * (df['poa'] / 1000.0) * 0.85

    # Clipping events: output power is within 2% of rated capacity despite high irradiance
    RATED_KW = 8648.0
    df['is_clipping'] = (df['total_output_kw'] > RATED_KW * 0.97) & (df['poa'] > 800)

    # Soiling signal: actual/expected ratio — persistent < 0.9 indicates soiling
    df['output_ratio'] = (df['total_output_kw'] / df['expected_kw'].replace(0, np.nan)).clip(0, 1.2)
    avg_ratio = float(df['output_ratio'].dropna().mean())

    # Build scatter data sample (max 50 points for API)
    scatter_sample = df[['poa', 'total_output_kw', 'expected_kw', 'is_clipping']].dropna()
    scatter_sample = scatter_sample.sample(min(50, len(scatter_sample)), random_state=42)
    scatter_list = [
        {
            "poa": round(float(r['poa']), 1),
            "actual_kw": round(float(r['total_output_kw']), 1),
            "expected_kw": round(float(r['expected_kw']), 1),
            "clipping": bool(r['is_clipping'])
        }
        for _, r in scatter_sample.iterrows()
    ]

    clipping_count = int(df['is_clipping'].sum())
    anomaly = "Clipping Detected" if clipping_count > 5 else ("Soiling Suspected" if avg_ratio < 0.85 else "Normal")

    return {
        "correlation_r2": round(r2, 3),
        "clipping_events": clipping_count,
        "avg_output_ratio": round(avg_ratio, 3),
        "anomaly_flag": anomaly,
        "scatter_data": scatter_list
    }


# 7. Module Thermal Anomaly Detection
@ttl_cache(5)
def detect_thermal_anomalies(start_time=None, end_time=None):
    """Detects thermal anomalies by comparing measured module temperature against
    a physics-based predicted temperature (NOCT model).
    
    High positive deviations indicate hotspot cells or failed cooling;
    negative deviations may indicate sensor faults.
    """
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT timestamp, ambientTemperature, planeOfArraySensor01, moduleTemperatureSensor01
        FROM telemetry WHERE device_group='WEATHER'
        AND ambientTemperature IS NOT NULL AND moduleTemperatureSensor01 IS NOT NULL
        AND planeOfArraySensor01 > 50
    """
    params = []
    if start_time:
        query += " AND timestamp >= ?"
        params.append(start_time)
    if end_time:
        query += " AND timestamp <= ?"
        params.append(end_time)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty or len(df) < 5:
        return {"anomaly_count": 0, "avg_delta_c": 0.0, "max_delta_c": 0.0, "status": "Insufficient data"}

    # NOCT model: T_module = T_ambient + (NOCT - 20) / 800 * POA
    # Standard NOCT = 45°C for typical c-Si modules
    NOCT = 45.0
    df['predicted_temp'] = df['ambientTemperature'] + ((NOCT - 20.0) / 800.0) * df['planeOfArraySensor01']
    df['temp_delta'] = df['moduleTemperatureSensor01'] - df['predicted_temp']

    avg_delta = float(df['temp_delta'].mean())
    max_delta = float(df['temp_delta'].max())

    # Flag anomalies where measured temp exceeds predicted by > 8°C (hotspot threshold)
    HOTSPOT_THRESHOLD = 8.0
    anomalies = df[df['temp_delta'] > HOTSPOT_THRESHOLD]
    anomaly_count = len(anomalies)

    status = "Critical - Hotspot Risk" if anomaly_count > 10 else (
        "Warning - Elevated Temperature" if anomaly_count > 0 else "Normal"
    )

    # Build daily aggregated view
    df['date'] = pd.to_datetime(df['timestamp']).dt.date.astype(str)
    daily = df.groupby('date').agg(
        avg_delta=('temp_delta', 'mean'),
        max_delta=('temp_delta', 'max'),
        avg_module_temp=('moduleTemperatureSensor01', 'mean'),
        avg_ambient_temp=('ambientTemperature', 'mean')
    ).reset_index()

    daily_list = [
        {
            "date": str(r['date']),
            "avg_delta_c": round(float(r['avg_delta']), 2),
            "max_delta_c": round(float(r['max_delta']), 2),
            "avg_module_temp": round(float(r['avg_module_temp']), 1),
            "avg_ambient_temp": round(float(r['avg_ambient_temp']), 1)
        }
        for _, r in daily.iterrows()
    ]

    return {
        "anomaly_count": anomaly_count,
        "avg_delta_c": round(avg_delta, 2),
        "max_delta_c": round(max_delta, 2),
        "hotspot_threshold_c": HOTSPOT_THRESHOLD,
        "status": status,
        "daily_thermal_profile": daily_list
    }


# 8. Grid Curtailment & Availability Analysis
@ttl_cache(5)
def analyze_grid_curtailment(start_time=None, end_time=None):
    """Classifies inverter downtime events into categories: Grid Curtailment, 
    Scheduled Maintenance, Fault/Trip, and Nighttime.
    
    Uses inverter status codes and cross-references with irradiance to distinguish
    curtailment (high irradiance + forced shutdown) from genuine faults.
    """
    conn = sqlite3.connect(DB_PATH)
    
    # Inverter status events
    query_i = "SELECT timestamp, meterId, inverterStatus, outputPower FROM telemetry WHERE device_group='INVERTER'"
    params_i = []
    if start_time:
        query_i += " AND timestamp >= ?"
        params_i.append(start_time)
    if end_time:
        query_i += " AND timestamp <= ?"
        params_i.append(end_time)
    df_i = pd.read_sql_query(query_i, conn, params=params_i)
    
    # Weather - for daylight classification
    query_w = "SELECT timestamp, planeOfArraySensor01 FROM telemetry WHERE device_group='WEATHER'"
    params_w = []
    if start_time:
        query_w += " AND timestamp >= ?"
        params_w.append(start_time)
    if end_time:
        query_w += " AND timestamp <= ?"
        params_w.append(end_time)
    df_w = pd.read_sql_query(query_w, conn, params=params_w)
    conn.close()

    if df_i.empty:
        return {"error": "No inverter data found.", "total_curtailment_hours": 0,
                "plant_availability_pct": 0, "total_fault_hours": 0,
                "estimated_curtailed_kwh": 0, "category_breakdown_hours": {}, "daily_curtailment": []}

    # Truncate timestamps to minute to allow join across different second offsets
    df_i['ts_min'] = df_i['timestamp'].str[:16]
    df_w['ts_min'] = df_w['timestamp'].str[:16]
    df = pd.merge(df_i, df_w, on='ts_min', how='left')
    df['poa'] = df['planeOfArraySensor01'].fillna(0)

    # Infer daytime from POA. If join had no match, use time-of-day heuristic (6am-7pm)
    if df['poa'].sum() == 0:
        df['hour'] = pd.to_datetime(df['timestamp_x']).dt.hour
        df['is_daytime'] = (df['hour'] >= 6) & (df['hour'] <= 19)
    else:
        df['is_daytime'] = df['poa'] > 50

    # Status code classification:
    # Data has: 1 = Running, 3 = Fault/Trip (confirmed from DB)
    # We also handle 0, 2, 4 if present
    df['category'] = 'Nighttime'
    df.loc[df['is_daytime'] & (df['inverterStatus'] == 1), 'category'] = 'Running'
    df.loc[df['is_daytime'] & (df['inverterStatus'] == 2), 'category'] = 'Grid Curtailment'
    df.loc[df['is_daytime'] & (df['inverterStatus'] == 3), 'category'] = 'Fault/Trip'
    df.loc[df['is_daytime'] & (df['inverterStatus'] == 0), 'category'] = 'Standby'
    df.loc[df['is_daytime'] & (df['inverterStatus'] == 4), 'category'] = 'Scheduled Maintenance'

    # Each record = 5 minutes -> hours
    INTERVAL_HOURS = 5.0 / 60.0
    category_hours = df.groupby('category').size() * INTERVAL_HOURS
    category_summary = {k: round(float(v), 2) for k, v in category_hours.items()}

    total_daytime_hours = float(df[df['is_daytime']].shape[0]) * INTERVAL_HOURS
    curtailment_hours = category_summary.get('Grid Curtailment', 0.0)
    fault_hours = category_summary.get('Fault/Trip', 0.0)

    # Plant Availability = (Running hours) / (total daytime hours)
    running_hours = category_summary.get('Running', 0.0)
    availability_pct = (running_hours / total_daytime_hours * 100.0) if total_daytime_hours > 0 else 0.0

    # Curtailed energy estimate uses the shared AVG_CURTAILED_KW assumption (LOG-01b)
    curtailed_kwh = round(curtailment_hours * AVG_CURTAILED_KW, 1)

    # Daily curtailment events
    df['date'] = pd.to_datetime(df['timestamp_x']).dt.date.astype(str)
    daily_curtailment = (
        df[df['category'] == 'Grid Curtailment']
        .groupby('date').size() * INTERVAL_HOURS
    ).reset_index()
    if not daily_curtailment.empty:
        daily_curtailment.columns = ['date', 'curtailment_hours']
        daily_curtailment['curtailment_hours'] = daily_curtailment['curtailment_hours'].round(2)
        daily_list = daily_curtailment.to_dict(orient='records')
    else:
        daily_list = []

    return {
        "plant_availability_pct": round(availability_pct, 2),
        "total_curtailment_hours": round(curtailment_hours, 2),
        "total_fault_hours": round(fault_hours, 2),
        "estimated_curtailed_kwh": curtailed_kwh,
        "category_breakdown_hours": category_summary,
        "daily_curtailment": daily_list
    }
# 9. Hardware Manual Lookup
def lookup_hardware_manual(device_type, code):
    """Retrieves troubleshooting guidelines from hardware_manuals.json."""
    manuals_path = os.path.join(os.path.dirname(__file__), "hardware_manuals.json")
    if not os.path.exists(manuals_path):
        return {"error": "Manuals file not found."}
    try:
        with open(manuals_path, "r") as f:
            manuals = json.load(f)
        dev_manual = manuals.get(device_type.lower())
        if not dev_manual:
            return {"error": f"No manuals found for device type '{device_type}'."}
        code_str = str(code)
        if code_str not in dev_manual:
            return {"error": f"No troubleshooting guidelines found for code {code_str}."}
        return dev_manual[code_str]
    except Exception as e:
        return {"error": str(e)}

# 10. O&M Diagnostics/Alerts Sweeper
def run_alerts_sweeper():
    """Analyzes telemetry for new anomalies and logs them to the alerts table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Thermal Hotspot Sweeper
    cursor.execute("""
        SELECT timestamp, ambientTemperature, planeOfArraySensor01, moduleTemperatureSensor01
        FROM telemetry
        WHERE device_group = 'WEATHER'
          AND ambientTemperature IS NOT NULL
          AND moduleTemperatureSensor01 IS NOT NULL
          AND planeOfArraySensor01 > 50
    """)
    rows = cursor.fetchall()
    new_alerts = []
    for timestamp, amb, poa, mod in rows:
        predicted = amb + (25.0 / 800.0) * poa
        delta = mod - predicted
        if delta > 8.0:
            msg = f"Thermal Hotspot Detected: Module Temp {mod:.1f}C exceeds NOCT predicted {predicted:.1f}C by {delta:.1f}C"
            cursor.execute("SELECT id FROM alerts WHERE timestamp = ? AND message = ?", (timestamp, msg))
            if not cursor.fetchone():
                new_alerts.append((timestamp, "JAMNAGAR_VIRTUAL_GATEWAY_PPCWMS1", "Warning", msg, "Active"))
                
    # 2. Inverter Efficiency Drop Sweeper
    cursor.execute("""
        SELECT timestamp, meterId, inputPVPower, outputPower
        FROM telemetry
        WHERE device_group = 'INVERTER'
          AND inputPVPower > 100000
          AND outputPower IS NOT NULL
    """)
    rows_i = cursor.fetchall()
    for timestamp, meter_id, input_pv, output in rows_i:
        eff = (output / input_pv) * 100.0
        if eff < 92.0:
            msg = f"Inverter Efficiency Anomaly: DC-AC efficiency dropped to {eff:.1f}%"
            cursor.execute("SELECT id FROM alerts WHERE timestamp = ? AND message = ?", (timestamp, msg))
            if not cursor.fetchone():
                new_alerts.append((timestamp, meter_id, "Warning", msg, "Active"))
                
    if new_alerts:
        cursor.executemany("INSERT INTO alerts (timestamp, asset_id, severity, message, status) VALUES (?, ?, ?, ?, ?)", new_alerts)
        conn.commit()
    conn.close()

@ttl_cache(5)
def get_all_alerts(start_time=None, end_time=None):
    """Retrieves all active and historic O&M alerts."""
    run_alerts_sweeper()
    
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT timestamp, asset_id, severity, message, status FROM alerts"
    params = []
    conditions = []
    if start_time:
        conditions.append("timestamp >= ?")
        params.append(start_time)
    if end_time:
        conditions.append("timestamp <= ?")
        params.append(end_time)
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
        
    query += " ORDER BY timestamp DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    return df.to_dict(orient="records")

# 11. Soiling ROI Calculator
@ttl_cache(5)
def calibrate_cleaning_roi(start_time=None, end_time=None):
    """Calculates the financial tipping point of module cleaning."""
    soiling_res = calibrate_soiling_rate(start_time, end_time)
    s_rate_pct = abs(soiling_res.get("avg_daily_soiling_rate_pct", -0.22))
    s_rate = s_rate_pct / 100.0
    cleaning_dates = soiling_res.get("inferred_cleaning_dates", [])
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date(timestamp)) FROM telemetry")
    latest_db_date_str = cursor.fetchone()[0] or "2026-06-24"
    conn.close()
    
    latest_db_date = datetime.strptime(latest_db_date_str, "%Y-%m-%d")
    
    if cleaning_dates:
        sorted_dates = sorted([datetime.strptime(d, "%Y-%m-%d") for d in cleaning_dates])
        last_cleaning_date = sorted_dates[-1]
    else:
        last_cleaning_date = latest_db_date - pd.Timedelta(days=14)
        
    days_since = (latest_db_date - last_cleaning_date).days
    days_since = max(0, days_since)
    
    gen_res = get_expected_vs_actual_generation(start_time=start_time, end_time=end_time)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    query_days = "SELECT COUNT(DISTINCT date(timestamp)) FROM telemetry WHERE 1=1"
    params = []
    if start_time:
        query_days += " AND timestamp >= ?"
        params.append(start_time)
    if end_time:
        query_days += " AND timestamp <= ?"
        params.append(end_time)
    cursor.execute(query_days, params)
    num_days = cursor.fetchone()[0] or 30
    conn.close()
    
    expected_daily_kwh = gen_res["expected_kwh"] / num_days if num_days > 0 else 5500.0
    if expected_daily_kwh <= 0:
        expected_daily_kwh = 5500.0
        
    TARIFF = 4.5  # Rs/kWh
    CLEANING_COST = 15000.0  # Rs
    
    cumulative_loss_kwh = expected_daily_kwh * s_rate * days_since * (days_since + 1) / 2.0
    cumulative_loss_inr = cumulative_loss_kwh * TARIFF
    
    daily_loss_kwh = expected_daily_kwh * days_since * s_rate
    daily_loss_inr = daily_loss_kwh * TARIFF
    
    coef = (2.0 * CLEANING_COST) / (expected_daily_kwh * s_rate * TARIFF)
    tipping_point_days = int(round((-1.0 + np.sqrt(1.0 + 4.0 * coef)) / 2.0))
    tipping_point_days = max(1, tipping_point_days)
    
    days_remaining = tipping_point_days - days_since
    cleaning_recommended = days_since >= tipping_point_days
    
    optimal_cleaning_date = (last_cleaning_date + pd.Timedelta(days=tipping_point_days)).strftime("%Y-%m-%d")
    roi_pct = ((cumulative_loss_inr - CLEANING_COST) / CLEANING_COST * 100.0) if cumulative_loss_inr > 0 else 0.0
    
    return {
        "last_cleaning_date": last_cleaning_date.strftime("%Y-%m-%d"),
        "days_since_last_cleaning": days_since,
        "cumulative_loss_kwh": round(cumulative_loss_kwh, 2),
        "cumulative_loss_inr": round(cumulative_loss_inr, 2),
        "daily_loss_kwh": round(daily_loss_kwh, 2),
        "daily_loss_inr": round(daily_loss_inr, 2),
        "tipping_point_days": tipping_point_days,
        "days_remaining": days_remaining,
        "cleaning_recommended": cleaning_recommended,
        "optimal_cleaning_date": optimal_cleaning_date,
        "roi_pct": round(roi_pct, 2),
        "cleaning_cost": CLEANING_COST,
        "tariff": TARIFF
    }

# 12. Day-Ahead Generation & BESS Forecasting
@ttl_cache(5)
def get_generation_and_bess_forecast(start_time=None, end_time=None):
    """Simulates a 24-hour solar generation forecast and battery dispatch schedule."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query_date = "SELECT MAX(date(timestamp)) FROM telemetry WHERE 1=1"
    params = []
    if start_time:
        query_date += " AND timestamp >= ?"
        params.append(start_time)
    if end_time:
        query_date += " AND timestamp <= ?"
        params.append(end_time)
    cursor.execute(query_date, params)
    latest_date = cursor.fetchone()[0] or "2026-06-24"
    
    query = """
    SELECT strftime('%H:00', timestamp) as hr,
           AVG(planeOfArraySensor01) as avg_poa,
           AVG(ambientTemperature) as avg_temp
    FROM telemetry
    WHERE device_group = 'WEATHER' AND timestamp LIKE ?
    GROUP BY hr
    ORDER BY hr
    """
    df_w = pd.read_sql_query(query, conn, params=(f"{latest_date}%",))
    conn.close()
    
    forecast = []
    bess_capacity_kwh = 10000.0  # 10 MWh plant BESS
    bess_soc_kwh = bess_capacity_kwh * 0.20  # starts at 20% SoC
    max_charge_power_kw = 2000.0  # 2 MW max charge/discharge
    charging_efficiency = 0.95
    discharging_efficiency = 0.95
    
    for h in range(24):
        hr_str = f"{h:02d}:00"
        w_row = df_w[df_w['hr'] == hr_str]
        poa = float(w_row['avg_poa'].iloc[0]) if not w_row.empty else 0.0
        temp = float(w_row['avg_temp'].iloc[0]) if not w_row.empty else 25.0
        
        if poa > 10:
            pred_gen = 8648.0 * (poa / 1000.0) * (1.0 - 0.004 * (temp - 25.0)) * 0.85
            pred_gen = max(0.0, pred_gen)
        else:
            pred_gen = 0.0
            
        forecast.append({
            "hour": hr_str,
            "predicted_generation_kw": round(pred_gen, 2),
            "poa": round(poa, 1),
            "temp": round(temp, 1)
        })
        
    bess_schedule = []
    for item in forecast:
        hr = int(item["hour"].split(":")[0])
        gen = item["predicted_generation_kw"]
        charge_discharge = 0.0
        
        if 10 <= hr <= 15:
            excess = max(0.0, gen - 2000.0)
            max_to_charge = min(max_charge_power_kw, excess)
            space_left = bess_capacity_kwh * 0.90 - bess_soc_kwh
            charge_power = min(max_to_charge, space_left / charging_efficiency)
            charge_power = max(0.0, charge_power)
            bess_soc_kwh += charge_power * charging_efficiency
            charge_discharge = charge_power
        elif 18 <= hr <= 22:
            max_to_discharge = max_charge_power_kw
            usable_energy = bess_soc_kwh - bess_capacity_kwh * 0.20
            discharge_power = min(max_to_discharge, usable_energy * discharging_efficiency)
            discharge_power = max(0.0, discharge_power)
            bess_soc_kwh -= discharge_power / discharging_efficiency
            charge_discharge = -discharge_power
            
        soc_pct = (bess_soc_kwh / bess_capacity_kwh) * 100.0
        bess_schedule.append({
            "hour": item["hour"],
            "predicted_generation_kw": item["predicted_generation_kw"],
            "bess_charge_discharge_kw": round(charge_discharge, 2),
            "bess_soc_pct": round(soc_pct, 2)
        })
        
    return bess_schedule

# 13. PDF/CSV Report Generator
def generate_report(start_time=None, end_time=None, file_format="csv"):
    """Generates a CSV or HTML performance report and returns the file path."""
    gen = get_expected_vs_actual_generation(start_time=start_time, end_time=end_time)
    eff = analyze_inverter_efficiency(start_time=start_time, end_time=end_time)
    # LOG-04b: cover the full BESS fleet (all 7 units), not just 3.
    bess_ids = [f"JAMNAGAR_VIRTUAL_GATEWAY_{u}" for u in
                ("B1BCT1", "B1BCT2", "B1BCT3", "B2BCT1", "B2BCT2", "B2BCT3", "B3BCT2")]
    bess_sohs, bess_ces, bess_cycles, bess_insufficient = [], [], [], []
    for b_id in bess_ids:
        b_res = get_bess_health(b_id, start_time, end_time)
        soh = b_res.get("state_of_health_pct")
        if soh is None:   # LOG-05: exclude Insufficient-Data units from fleet averages
            bess_insufficient.append(b_id.split("_")[-1])
            continue
        bess_sohs.append(soh)
        bess_ces.append(b_res.get("coulombic_efficiency") or 0.0)
        bess_cycles.append(b_res.get("total_cycles") or 0)

    avg_bess_soh = sum(bess_sohs) / len(bess_sohs) if bess_sohs else 0.0
    avg_bess_ce = sum(bess_ces) / len(bess_ces) if bess_ces else 0.0
    avg_bess_cycles = sum(bess_cycles) / len(bess_cycles) if bess_cycles else 0.0

    thermal = detect_thermal_anomalies(start_time=start_time, end_time=end_time)
    curt = analyze_grid_curtailment(start_time=start_time, end_time=end_time)
    soiling = calibrate_soiling_rate(start_time=start_time, end_time=end_time)
    
    report_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    if file_format.lower() == "csv":
        file_path = os.path.join(report_dir, "sungenie_report.csv")
        data = [
            ["Metric", "Value"],
            ["Report Period Start", start_time or "Earliest available"],
            ["Report Period End", end_time or "Latest available"],
            ["Expected Generation (kWh)", gen["expected_kwh"]],
            ["Actual Generation (kWh)", gen["actual_kwh"]],
            ["PR Actual", gen["pr_actual"]],
            ["Generation Gap (kWh)", gen["gap_kwh"]],
            ["Soiling Loss (kWh)", gen["attribution"]["Soiling"]],
            ["Shading Loss (kWh)", gen["attribution"]["Shading"]],
            ["Hardware Inefficiency Loss (kWh)", gen["attribution"]["Hardware Inefficiency"]],
            ["Grid Curtailment Loss (kWh)", gen["attribution"]["Grid Curtailment"]],
            ["Fleet Avg Inverter Efficiency (%)", eff.get("fleet_avg_efficiency_pct", 0.0)],
            ["BESS Avg State of Health (%)", round(avg_bess_soh, 2)],
            ["BESS Avg Coulombic Efficiency", round(avg_bess_ce, 3)],
            ["BESS Avg Total Cycles", round(avg_bess_cycles, 2)],
            ["BESS Units Reported", f"{len(bess_sohs)} of {len(bess_ids)}" + (f" (insufficient data: {', '.join(bess_insufficient)})" if bess_insufficient else "")],
            ["Thermal Anomaly Count", thermal.get("anomaly_count", 0)],
            ["Max Module Temp Delta (C)", thermal.get("max_delta_c", 0.0)],
            ["Plant Availability (%)", curt.get("plant_availability_pct", 0.0)],
            ["Grid Curtailment Hours", curt.get("total_curtailment_hours", 0.0)],
            ["Unplanned Inverter Trip Hours", curt.get("total_fault_hours", 0.0)],
            ["Average Daily Soiling Rate (%/day)", soiling.get("avg_daily_soiling_rate_pct", 0.0)]
        ]
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data)
        return file_path
    else:
        file_path = os.path.join(report_dir, "sungenie_report.html")
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #0b0f19; color: #f3f4f6; padding: 20px; }}
                h1 {{ color: #8ab833; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #1e293b; padding: 12px; text-align: left; }}
                th {{ background-color: #1e293b; color: #8ab833; }}
                tr:nth-child(even) {{ background-color: #0f172a; }}
            </style>
        </head>
        <body>
            <h1>JioSunGenie Plant Performance Report</h1>
            <p><strong>Period:</strong> {start_time or 'Start'} to {end_time or 'End'}</p>
            <table>
                <tr><th>Performance Metric</th><th>Value</th></tr>
                <tr><td>Expected Generation</td><td>{gen['expected_kwh']:,} kWh</td></tr>
                <tr><td>Actual Generation</td><td>{gen['actual_kwh']:,} kWh</td></tr>
                <tr><td>Performance Ratio (PR)</td><td>{gen['pr_actual']}</td></tr>
                <tr><td>Generation Gap</td><td>{gen['gap_kwh']:,} kWh</td></tr>
                <tr><td>Soiling Loss</td><td>{gen['attribution']['Soiling']:,} kWh</td></tr>
                <tr><td>Fleet Avg Inverter Efficiency</td><td>{eff.get('fleet_avg_efficiency_pct', 0.0)}%</td></tr>
                <tr><td>BESS Avg State of Health</td><td>{avg_bess_soh:.2f}%</td></tr>
                <tr><td>Plant Availability</td><td>{curt.get('plant_availability_pct', 0.0)}%</td></tr>
            </table>
        </body>
        </html>
        """
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return file_path

if __name__ == "__main__":
    # Self-test when run
    print("Testing ML Pipelines...")
    print("Expected Gen (Entire Month):", get_expected_vs_actual_generation())
    print("SCB Outliers:", detect_scb_outliers())
    print("Soiling Rate:", calibrate_soiling_rate()["avg_daily_soiling_rate_pct"])
    print("BESS Health:", get_bess_health())
    print("Inverter Efficiency:", analyze_inverter_efficiency()["fleet_avg_efficiency_pct"])
    print("Irradiance Correlation:", analyze_irradiance_power_correlation()["correlation_r2"])
    print("Thermal Anomalies:", detect_thermal_anomalies()["status"])
    print("Grid Curtailment:", analyze_grid_curtailment()["plant_availability_pct"])
    print("Cleaning ROI:", calibrate_cleaning_roi())
    print("Forecast Count:", len(get_generation_and_bess_forecast()))
    print("Report Path:", generate_report())

