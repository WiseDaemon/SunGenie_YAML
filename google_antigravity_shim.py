import os
import sys
import asyncio
import json
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv(dotenv_path="C:/LLM/.env")

sys.path.append(os.path.dirname(__file__))
import ml_pipelines
import agent_setup

MODEL_NAME = "gemma-4-26b-a4b-it"

# Nvidia API credentials
ISING_KEY  = "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"
LLAMA_KEY  = "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"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


class LocalAgentConfig:
    def __init__(self, tools=None, system_instructions=None, **kwargs):
        self.tools = tools or []
        self.system_instructions = system_instructions
        self.kwargs = kwargs


class AgentResponse:
    """Holds text, streaming thoughts, and an optional chart specification."""

    def __init__(self, text_content, thoughts=None, chart=None):
        self._text   = text_content
        self._thoughts = thoughts or ["Processing response..."]
        self.chart   = chart  # dict with Chart.js-compatible config, or None

    async def text(self):
        return self._text

    def __aiter__(self):
        class TokenIterator:
            def __init__(self, text):
                self.words = text.split(" ")
                self.idx = 0
            async def __anext__(self):
                if self.idx >= len(self.words):
                    raise StopAsyncIteration
                val = self.words[self.idx] + " "
                self.idx += 1
                await asyncio.sleep(0.01)
                return val
        return TokenIterator(self._text)

    @property
    def thoughts(self):
        class ThoughtIterator:
            def __init__(self, t):
                self.t = t
                self.idx = 0
            def __aiter__(self):
                return self
            async def __anext__(self):
                if self.idx >= len(self.t):
                    raise StopAsyncIteration
                val = self.t[self.idx] + "\n"
                self.idx += 1
                await asyncio.sleep(0.1)
                return val
        return ThoughtIterator(self._thoughts)


# ── Chart spec builders ────────────────────────────────────────────────────────
# Each returns a Chart.js-compatible config dict (passed as JSON to the front-end)

def _chart_pr_gap(gen_res):
    attr = gen_res["attribution"]
    return {
        "type": "bar",
        "title": f"PR Gap Attribution — {gen_res['gap_kwh']:,} kWh total gap",
        "description": "Calculates expected vs actual generation and attributes the gap to soiling, shading, hardware inefficiency, and grid curtailment.",
        "data": {
            "labels": list(attr.keys()),
            "datasets": [{
                "label": "Loss (kWh)",
                "data": list(attr.values()),
                "backgroundColor": ["rgba(138,184,51,0.75)", "rgba(9,137,177,0.75)",
                                    "rgba(74,181,196,0.75)", "rgba(2,150,118,0.75)"],
                "borderRadius": 6
            }]
        },
        "options": {"scales": {"y": {"beginAtZero": True}}}
    }


def _chart_scb(outlier_res):
    all_strings = {**outlier_res.get("underperforming_strings", {})}
    mean = outlier_res.get("mean_current", 0)
    std  = outlier_res.get("std_dev", 0)
    # Build labels + z-scores for outlier chart
    labels = list(all_strings.keys()) or [f"SCB {i}" for i in range(1, 11)]
    zscores = [v["z_score"] for v in all_strings.values()] if all_strings else [0] * 10
    colors = ["rgba(239,68,68,0.75)" if z < -2 else "rgba(138,184,51,0.65)" for z in zscores]
    return {
        "type": "bar",
        "title": f"SCB String Z-Scores — Mean {mean}A, σ {std}A",
        "description": "Computes Z-scores of String Combiner Box (SCB) currents to detect underperforming strings (Z-score < -2.0) relative to the inverter average.",
        "data": {
            "labels": labels,
            "datasets": [{
                "label": "Z-Score",
                "data": zscores,
                "backgroundColor": colors,
                "borderRadius": 4
            }]
        },
        "options": {"scales": {"y": {"title": {"display": True, "text": "Z-Score"}}}}
    }


def _chart_bess(bess_res):
    soh = bess_res.get("state_of_health_pct", 0)
    return {
        "type": "doughnut",
        "title": f"BESS State of Health — {bess_res.get('bess_id','').split('_')[-1]}",
        "description": "Visualizes BESS State of Health (SoH) as healthy vs degraded capacity based on Coulombic efficiency and charge cycle counts.",
        "data": {
            "labels": ["Healthy Capacity", "Degraded"],
            "datasets": [{
                "data": [round(soh, 2), round(100.0 - soh, 2)],
                "backgroundColor": ["rgba(138,184,51,0.8)", "rgba(60,70,100,0.5)"],
                "borderWidth": 1,
                "borderColor": "rgba(11,15,25,0.5)"
            }]
        },
        "options": {"cutout": "70%"}
    }


def _chart_bess_comparison(bess_results):
    labels = [k.split("_")[-1] for k in bess_results.keys()]
    sohs = [v.get("state_of_health_pct", 0) for v in bess_results.values()]
    return {
        "type": "bar",
        "title": "BESS State of Health Comparison (%)",
        "description": "Compares State of Health (SoH) across multiple Battery Capacity Tracker (BCT) units.",
        "data": {
            "labels": labels,
            "datasets": [{
                "label": "SoH %",
                "data": sohs,
                "backgroundColor": ["rgba(138,184,51,0.75)", "rgba(9,137,177,0.75)", "rgba(74,181,196,0.75)"],
                "borderRadius": 4
            }]
        },
        "options": {"scales": {"y": {"min": 90, "max": 100}}}
    }


def _chart_mfm_trend(mfm_results):
    asset = list(mfm_results.keys())[0]
    records = mfm_results[asset]
    if not records: return None
    records = list(reversed(records))
    return {
        "type": "line",
        "title": f"Active Power Trend — {asset.split('_')[-1]}",
        "description": "Plots the active power (kW) trend over recent time intervals for the selected MFM meter.",
        "data": {
            "labels": [r["timestamp"][11:16] for r in records],
            "datasets": [{
                "label": "Active Power (kW)",
                "data": [r["activePower"] for r in records],
                "borderColor": "#0989b1",
                "backgroundColor": "rgba(9,137,177,0.12)",
                "fill": True,
                "tension": 0.4
            }]
        }
    }


def _chart_efficiency(eff_res):
    curve = eff_res.get("fleet_curve", [])
    return {
        "type": "line",
        "title": f"Inverter DC-AC Efficiency Curve — Fleet Avg {eff_res.get('fleet_avg_efficiency_pct', 0)}%",
        "description": "Aggregates inverter DC-AC conversion efficiency across different load factor bins (0-10% to 90-100%).",
        "data": {
            "labels": [p["load_bin"] for p in curve],
            "datasets": [{
                "label": "Avg Efficiency %",
                "data": [p["avg_efficiency_pct"] for p in curve],
                "borderColor": "#4ab5c4",
                "backgroundColor": "rgba(74,181,196,0.12)",
                "fill": True,
                "tension": 0.4,
                "pointRadius": 5
            }, {
                "label": "Target (96%)",
                "data": [96] * len(curve),
                "borderColor": "rgba(239,68,68,0.5)",
                "borderDash": [5, 4],
                "borderWidth": 1.5,
                "pointRadius": 0,
                "fill": False
            }]
        },
        "options": {"scales": {"y": {"min": 85, "max": 100,
                    "title": {"display": True, "text": "Efficiency (%)"}},
                    "x": {"title": {"display": True, "text": "Load Factor Bin"}}}}
    }


def _chart_irradiance(irr_res):
    scatter = irr_res.get("scatter_data", [])
    normal   = [{"x": d["poa"], "y": d["actual_kw"]}  for d in scatter if not d["clipping"]]
    clipping = [{"x": d["poa"], "y": d["actual_kw"]}  for d in scatter if d["clipping"]]
    expected = sorted([{"x": d["poa"], "y": d["expected_kw"]} for d in scatter], key=lambda p: p["x"])
    return {
        "type": "scatter",
        "title": f"Irradiance vs Power — R²={irr_res.get('correlation_r2',0)} | {irr_res.get('anomaly_flag','')}",
        "description": "Plots Plane-of-Array (POA) irradiance against plant output power. Red dots show inverter power clipping; dashed line shows linear target.",
        "data": {
            "datasets": [
                {"label": "Actual Output", "data": normal,
                 "backgroundColor": "rgba(138,184,51,0.65)", "pointRadius": 4},
                {"label": "Clipping Events", "data": clipping,
                 "backgroundColor": "rgba(239,68,68,0.8)", "pointRadius": 6},
                {"label": "Expected (PR=0.85)", "data": expected, "type": "line",
                 "borderColor": "rgba(74,181,196,0.7)", "borderDash": [4, 3],
                 "borderWidth": 1.5, "pointRadius": 0, "fill": False}
            ]
        },
        "options": {"scales": {
            "x": {"title": {"display": True, "text": "POA Irradiance (W/m²)"}},
            "y": {"title": {"display": True, "text": "Output Power (kW)"}}
        }}
    }


def _chart_thermal(thermal_res):
    profile = thermal_res.get("daily_thermal_profile", [])
    return {
        "type": "line",
        "title": f"Module Thermal Profile — {thermal_res.get('anomaly_count',0)} Hotspot Events",
        "description": "Tracks daily module temperature delta profiles over time against the standard NOCT threshold to flag potential hotspot risks.",
        "data": {
            "labels": [p["date"] for p in profile],
            "datasets": [{
                "label": "Avg Temp Delta (°C)",
                "data": [p["avg_delta_c"] for p in profile],
                "borderColor": "#f59e0b",
                "backgroundColor": "rgba(245,158,11,0.12)",
                "fill": True, "tension": 0.4, "pointRadius": 4
            }, {
                "label": "Hotspot Threshold (8°C)",
                "data": [8] * len(profile),
                "borderColor": "rgba(239,68,68,0.6)",
                "borderDash": [5, 4], "borderWidth": 1.5,
                "pointRadius": 0, "fill": False
            }]
        },
        "options": {"scales": {
            "y": {"title": {"display": True, "text": "Temp Delta (°C)"}},
            "x": {"title": {"display": True, "text": "Date"}}
        }}
    }


def _chart_curtailment(curt_res):
    cats = curt_res.get("category_breakdown_hours", {})
    COLOR_MAP = {
        "Running": "rgba(138,184,51,0.8)",
        "Grid Curtailment": "rgba(245,158,11,0.8)",
        "Fault/Trip": "rgba(239,68,68,0.8)",
        "Nighttime": "rgba(60,70,100,0.7)",
        "Standby": "rgba(74,181,196,0.6)",
        "Scheduled Maintenance": "rgba(139,92,246,0.7)"
    }
    labels = list(cats.keys())
    return {
        "type": "doughnut",
        "title": f"Operating State Breakdown — Availability {curt_res.get('plant_availability_pct',0)}%",
        "description": "Categorizes the plant's operating states (Running, Grid Curtailment, Fault/Trip, Nighttime, Standby, Maintenance) during daylight hours.",
        "data": {
            "labels": labels,
            "datasets": [{
                "data": list(cats.values()),
                "backgroundColor": [COLOR_MAP.get(l, "rgba(150,150,150,0.6)") for l in labels],
                "borderWidth": 1, "borderColor": "rgba(11,15,25,0.5)", "hoverOffset": 8
            }]
        },
        "options": {"cutout": "60%", "plugins": {"legend": {"position": "right"}}}
    }


def _chart_comparison(label_val_pairs, title, dataset_label):
    labels = [p[0] for p in label_val_pairs]
    values = [p[1] for p in label_val_pairs]
    return {
        "type": "bar",
        "title": title,
        "description": "Compares normalized telemetry values across multiple assets.",
        "data": {
            "labels": labels,
            "datasets": [{
                "label": dataset_label,
                "data": values,
                "backgroundColor": ["rgba(138,184,51,0.75)", "rgba(9,137,177,0.75)", "rgba(74,181,196,0.75)", "rgba(2,150,118,0.75)"],
                "borderRadius": 4
            }]
        },
        "options": {"scales": {"y": {"beginAtZero": True}}}
    }


def get_generic_asset_data(meter_id, limit=5):
    import sqlite3
    conn = sqlite3.connect(ml_pipelines.DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM telemetry WHERE meterId = ? ORDER BY timestamp DESC LIMIT {limit}", (meter_id,))
    cols = [col[0] for col in cursor.description]
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return []
    non_null_cols = []
    for i, col in enumerate(cols):
        if any(row[i] is not None for row in rows):
            non_null_cols.append((col, i))
    res = []
    for row in rows:
        item = {}
        for col, idx in non_null_cols:
            item[col] = row[idx]
        res.append(item)
    return res


# ── Agent class ────────────────────────────────────────────────────────────────

class Agent:
    def __init__(self, config):
        self.config = config
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        genai.configure(api_key=api_key)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def chat(self, prompt: str) -> AgentResponse:
        loop = asyncio.get_running_loop()
        prompt_lower = prompt.lower()
        context_data = ""
        chart_spec   = None
        thoughts     = ["Initializing local telemetry diagnostics..."]
        use_ising    = False  # True → Ising 35B specialist; False → Llama 8B

        # ── Direct SQL query — check FIRST to avoid asset hijacking ──────────────────
        if "sql" in prompt_lower and "weather" in prompt_lower:
            thoughts.append("Executing SQL query for weather telemetry...")
            default_query = "SELECT timestamp, ambientTemperature, planeOfArraySensor01 FROM telemetry WHERE device_group='WEATHER' ORDER BY timestamp DESC LIMIT 5"
            res_str = agent_setup.execute_sql_query(default_query)
            return AgentResponse(f"**SQL Query Results (Weather Data):**\n```json\n{res_str}\n```", thoughts)

        elif "select" in prompt_lower and "from" in prompt_lower:
            thoughts.append("Executing SQL query locally on SQLite database...")
            res_str = agent_setup.execute_sql_query(prompt)
            return AgentResponse(f"**SQL Query Results:**\n```json\n{res_str}\n```", thoughts)

        # ── Dynamic Asset extractor & resolver ─────────────────────────────────
        import re
        db_assets = [
            "B1BCT1", "B1BCT2", "B1BCT3", "B2BCT1", "B2BCT2", "B2BCT3", "B3BCT2",
            "B1INV1", "B3INV1",
            "B1MFM1", "B1MFM2", "B3MFM1", "B3MFM2", "B3MFM3",
            "B1PCS1", "B2PCS1", "B3PCS1",
            "B1PQM2", "B2PQM1", "B3PQM1", "B3PQM2",
            "B2DCCON1", "B2DCCON2", "B2DCCON3",
            "PPCWMS1"
        ]
        for i in range(1, 17):
            db_assets.append(f"B2MFM{i}")

        p_upper = prompt.upper()
        found_assets = []

        def add_asset(asset_short):
            full_name = f"JAMNAGAR_VIRTUAL_GATEWAY_{asset_short}"
            if asset_short == "PPCWMS1":
                full_name = "JAMNAGAR_VIRTUAL_GATEWAY_PPCWMS1"
            if full_name not in found_assets:
                found_assets.append(full_name)

        # 1. Exact or normalized matching of full short names (e.g. B2BCT3, B2_BCT3)
        clean_p = p_upper
        clean_p = re.sub(r'\bBLOCK\s*[-_]?\s*(\d+)\b', r'B\1', clean_p)
        clean_p = re.sub(r'[^A-Z0-9]', '', clean_p)
        
        sorted_db_assets = sorted(db_assets, key=lambda x: (-len(x), x))
        temp_p = clean_p
        matched_exact = []
        for asset in sorted_db_assets:
            if asset in temp_p:
                matched_exact.append(asset)
                temp_p = temp_p.replace(asset, "____")
        
        for asset in matched_exact:
            add_asset(asset)

        # 2. Match type + number pairs without block (e.g. MFM 12)
        type_synonyms = {
            "BCT": ["BCT", "BATTERY", "BATTERIES", "BESS"],
            "INV": ["INV", "INVERTER", "INVERTERS"],
            "MFM": ["MFM", "METER", "METERS"],
            "PCS": ["PCS"],
            "PQM": ["PQM"],
            "DCCON": ["DCCON", "CONVERTER", "CONVERTERS"]
        }
        
        norm_p = p_upper
        norm_p = re.sub(r'\bBLOCK\s*[-_]?\s*(\d+)\b', r'B\1', norm_p)
        
        for std_type, syns in type_synonyms.items():
            for syn in syns:
                pattern = rf'\b{syn}\s*[-_]?\s*(\d+)\b'
                matches = re.findall(pattern, norm_p)
                for m in matches:
                    num = int(m)
                    candidates = []
                    for asset in db_assets:
                        if asset == "PPCWMS1":
                            continue
                        m_parts = re.match(r"(B\d+)(INV|BCT|MFM|PCS|PQM|DCCON)(\d+)", asset)
                        if m_parts:
                            b, t, n = m_parts.groups()
                            if t == std_type and int(n) == num:
                                candidates.append((asset, b))
                    
                    block_match = None
                    if "B1" in norm_p or "BLOCK 1" in norm_p or "BLOCK-1" in norm_p:
                        block_match = "B1"
                    elif "B2" in norm_p or "BLOCK 2" in norm_p or "BLOCK-2" in norm_p:
                        block_match = "B2"
                    elif "B3" in norm_p or "BLOCK 3" in norm_p or "BLOCK-3" in norm_p:
                        block_match = "B3"
                        
                    if block_match:
                        filtered = [cand for cand, blk in candidates if blk == block_match]
                        if filtered:
                            for f in filtered:
                                add_asset(f)
                            continue
                    
                    for cand, blk in candidates:
                        add_asset(cand)

        # 3. Weather station references
        if any(w in p_upper for w in ["WEATHER", "WMS", "PPCWMS1", "PPCWMS"]):
            add_asset("PPCWMS1")

        # Check if they want to compare all assets of a certain type
        compare_all_type = None
        if "compare" in prompt_lower or "versus" in prompt_lower or " vs " in prompt_lower or "difference between" in prompt_lower:
            if "inverter" in prompt_lower:
                compare_all_type = "INVERTER"
            elif "battery" in prompt_lower or "bess" in prompt_lower or "bct" in prompt_lower:
                compare_all_type = "BESS"
            elif "meter" in prompt_lower or "mfm" in prompt_lower:
                compare_all_type = "METER"
            elif "pcs" in prompt_lower:
                compare_all_type = "PCS"
            elif "dccon" in prompt_lower or "converter" in prompt_lower:
                compare_all_type = "DCCON"
            elif "pqm" in prompt_lower:
                compare_all_type = "PQM"
                
        if not found_assets and compare_all_type:
            if compare_all_type == "INVERTER":
                found_assets = ["JAMNAGAR_VIRTUAL_GATEWAY_B1INV1", "JAMNAGAR_VIRTUAL_GATEWAY_B3INV1"]
            elif compare_all_type == "BESS":
                found_assets = [f"JAMNAGAR_VIRTUAL_GATEWAY_B1BCT{i}" for i in range(1, 4)] + [f"JAMNAGAR_VIRTUAL_GATEWAY_B2BCT{i}" for i in range(1, 4)] + ["JAMNAGAR_VIRTUAL_GATEWAY_B3BCT2"]
            elif compare_all_type == "METER":
                found_assets = ["JAMNAGAR_VIRTUAL_GATEWAY_B1MFM1", "JAMNAGAR_VIRTUAL_GATEWAY_B1MFM2"] + [f"JAMNAGAR_VIRTUAL_GATEWAY_B2MFM{i}" for i in range(1, 17)] + ["JAMNAGAR_VIRTUAL_GATEWAY_B3MFM1", "JAMNAGAR_VIRTUAL_GATEWAY_B3MFM2", "JAMNAGAR_VIRTUAL_GATEWAY_B3MFM3"]
            elif compare_all_type == "PCS":
                found_assets = ["JAMNAGAR_VIRTUAL_GATEWAY_B1PCS1", "JAMNAGAR_VIRTUAL_GATEWAY_B2PCS1", "JAMNAGAR_VIRTUAL_GATEWAY_B3PCS1"]
            elif compare_all_type == "DCCON":
                found_assets = ["JAMNAGAR_VIRTUAL_GATEWAY_B2DCCON1", "JAMNAGAR_VIRTUAL_GATEWAY_B2DCCON2", "JAMNAGAR_VIRTUAL_GATEWAY_B2DCCON3"]
            elif compare_all_type == "PQM":
                found_assets = ["JAMNAGAR_VIRTUAL_GATEWAY_B1PQM2", "JAMNAGAR_VIRTUAL_GATEWAY_B2PQM1", "JAMNAGAR_VIRTUAL_GATEWAY_B3PQM1", "JAMNAGAR_VIRTUAL_GATEWAY_B3PQM2"]

        # Check if they want to list assets
        is_list_request = (
            any(k in prompt_lower for k in ["list", "show", "available", "inventory", "catalog", "catalog of"]) or
            ("what" in prompt_lower and "available" in prompt_lower) or
            ("which" in prompt_lower and "available" in prompt_lower) or
            ("what are the" in prompt_lower and any(t in prompt_lower for t in ["batter", "inverter", "meter", "pcs", "pqm", "dccon", "wms", "asset", "device"]))
        )
        if not found_assets and is_list_request and any(k in prompt_lower for k in ["asset", "device", "inventory", "station", "meter", "battery", "inverter", "pcs", "pqm", "dccon", "wms", "bess", "bct", "converter"]):
            if "battery" in prompt_lower or "bess" in prompt_lower or "bct" in prompt_lower:
                thoughts.append("Listing available BESS assets...")
                bess_list = [f"- `JAMNAGAR_VIRTUAL_GATEWAY_B1BCT{i}` (B1BCT{i})" for i in range(1, 4)] + [f"- `JAMNAGAR_VIRTUAL_GATEWAY_B2BCT{i}` (B2BCT{i})" for i in range(1, 4)] + ["- `JAMNAGAR_VIRTUAL_GATEWAY_B3BCT2` (B3BCT2)"]
                return AgentResponse("### Active BESS (Battery Capacity Tracker) Assets:\n" + "\n".join(bess_list), thoughts)
            elif "inverter" in prompt_lower:
                thoughts.append("Listing active Inverter assets...")
                return AgentResponse("### Active Inverter Assets:\n- `JAMNAGAR_VIRTUAL_GATEWAY_B1INV1` (B1INV1 - Block 1)\n- `JAMNAGAR_VIRTUAL_GATEWAY_B3INV1` (B3INV1 - Block 3)", thoughts)
            elif "meter" in prompt_lower or "mfm" in prompt_lower:
                thoughts.append("Listing MFM meter assets...")
                meters = ["- `JAMNAGAR_VIRTUAL_GATEWAY_B1MFM1` (B1MFM1)", "- `JAMNAGAR_VIRTUAL_GATEWAY_B1MFM2` (B1MFM2)"] + [f"- `JAMNAGAR_VIRTUAL_GATEWAY_B2MFM{i}` (B2MFM{i})" for i in range(1, 17)] + ["- `JAMNAGAR_VIRTUAL_GATEWAY_B3MFM1` (B3MFM1)", "- `JAMNAGAR_VIRTUAL_GATEWAY_B3MFM2` (B3MFM2)", "- `JAMNAGAR_VIRTUAL_GATEWAY_B3MFM3` (B3MFM3)"]
                return AgentResponse("### Active MFM Meter Assets:\n" + "\n".join(meters), thoughts)
            elif "pcs" in prompt_lower:
                return AgentResponse("### Active PCS (Power Conversion System) Assets:\n- `JAMNAGAR_VIRTUAL_GATEWAY_B1PCS1` (B1PCS1)\n- `JAMNAGAR_VIRTUAL_GATEWAY_B2PCS1` (B2PCS1)\n- `JAMNAGAR_VIRTUAL_GATEWAY_B3PCS1` (B3PCS1)", thoughts)
            elif "pqm" in prompt_lower:
                return AgentResponse("### Active PQM (Power Quality Monitor) Assets:\n- `JAMNAGAR_VIRTUAL_GATEWAY_B1PQM2` (B1PQM2)\n- `JAMNAGAR_VIRTUAL_GATEWAY_B2PQM1` (B2PQM1)\n- `JAMNAGAR_VIRTUAL_GATEWAY_B3PQM1` (B3PQM1)\n- `JAMNAGAR_VIRTUAL_GATEWAY_B3PQM2` (B3PQM2)", thoughts)
            elif "dccon" in prompt_lower or "converter" in prompt_lower:
                return AgentResponse("### Active DC Converter Assets:\n- `JAMNAGAR_VIRTUAL_GATEWAY_B2DCCON1` (B2DCCON1)\n- `JAMNAGAR_VIRTUAL_GATEWAY_B2DCCON2` (B2DCCON2)\n- `JAMNAGAR_VIRTUAL_GATEWAY_B2DCCON3` (B2DCCON3)", thoughts)
            elif "weather" in prompt_lower or "wms" in prompt_lower:
                return AgentResponse("### Active Weather Monitoring Station:\n- `JAMNAGAR_VIRTUAL_GATEWAY_PPCWMS1` (PPCWMS1)", thoughts)
            else:
                thoughts.append("Compiling structured inventory of all plant assets...")
                bess_list_items = [f"   - `JAMNAGAR_VIRTUAL_GATEWAY_B1BCT{i}` (B1BCT{i})" for i in range(1, 4)] + [f"   - `JAMNAGAR_VIRTUAL_GATEWAY_B2BCT{i}` (B2BCT{i})" for i in range(1, 4)] + ["   - `JAMNAGAR_VIRTUAL_GATEWAY_B3BCT2` (B3BCT2)"]
                meters_list_items = [f"   - `JAMNAGAR_VIRTUAL_GATEWAY_B1MFM1` (B1MFM1)", f"   - `JAMNAGAR_VIRTUAL_GATEWAY_B1MFM2` (B1MFM2)"] + [f"   - `JAMNAGAR_VIRTUAL_GATEWAY_B2MFM{i}` (B2MFM{i})" for i in range(1, 17)] + [f"   - `JAMNAGAR_VIRTUAL_GATEWAY_B3MFM{i}` (B3MFM{i})" for i in range(1, 4)]
                bess_str = "\n".join(bess_list_items)
                meters_str = "\n".join(meters_list_items)
                inventory = (
                    "### Jamnagar Solar Facility — Asset Inventory\n\n"
                    "The facility is monitored across 8 device groups containing **42 active assets**:\n\n"
                    "1. **WEATHER (WMS)** (1 asset):\n"
                    "   - `JAMNAGAR_VIRTUAL_GATEWAY_PPCWMS1` (PPCWMS1)\n\n"
                    "2. **INVERTERS** (2 assets):\n"
                    "   - `JAMNAGAR_VIRTUAL_GATEWAY_B1INV1` (B1INV1)\n"
                    "   - `JAMNAGAR_VIRTUAL_GATEWAY_B3INV1` (B3INV1)\n\n"
                    "3. **BESS (Battery Storage)** (7 assets):\n"
                    f"{bess_str}\n\n"
                    "4. **METERS (MFM)** (21 assets):\n"
                    f"{meters_str}\n\n"
                    "5. **PCS (Power Conversion System)** (3 assets):\n"
                    "   - `JAMNAGAR_VIRTUAL_GATEWAY_B1PCS1` (B1PCS1)\n"
                    "   - `JAMNAGAR_VIRTUAL_GATEWAY_B2PCS1` (B2PCS1)\n"
                    "   - `JAMNAGAR_VIRTUAL_GATEWAY_B3PCS1` (B3PCS1)\n\n"
                    "6. **PQM (Power Quality Monitor)** (4 assets):\n"
                    "   - `JAMNAGAR_VIRTUAL_GATEWAY_B1PQM2` (B1PQM2)\n"
                    "   - `JAMNAGAR_VIRTUAL_GATEWAY_B2PQM1` (B2PQM1)\n"
                    "   - `JAMNAGAR_VIRTUAL_GATEWAY_B3PQM1` (B3PQM1)\n"
                    "   - `JAMNAGAR_VIRTUAL_GATEWAY_B3PQM2` (B3PQM2)\n\n"
                    "7. **DC CONVERTERS (DCCON)** (3 assets):\n"
                    "   - `JAMNAGAR_VIRTUAL_GATEWAY_B2DCCON1` (B2DCCON1)\n"
                    "   - `JAMNAGAR_VIRTUAL_GATEWAY_B2DCCON2` (B2DCCON2)\n"
                    "   - `JAMNAGAR_VIRTUAL_GATEWAY_B2DCCON3` (B2DCCON3)\n\n"
                    "8. **GATEWAYS** (1 asset):\n"
                    "   - `JAMNAGAR_VIRTUAL_GATEWAY`\n\n"
                    "You can query insights, telemetry trends, or compare any specific assets (e.g. *'compare B2BCT3 and B2BCT1'* or *'active power of B1PCS1'*)."
                )
                return AgentResponse(inventory, thoughts)

        # ── Intent classifier ──────────────────────────────────────────────────

        # 0. Asset-specific routing
        if found_assets:
            bess_assets = [a for a in found_assets if "BCT" in a]
            mfm_assets = [a for a in found_assets if "MFM" in a]
            inv_assets = [a for a in found_assets if "INV" in a]
            weather_assets = [a for a in found_assets if "WMS" in a]
            pcs_assets = [a for a in found_assets if "PCS" in a]
            pqm_assets = [a for a in found_assets if "PQM" in a]
            dccon_assets = [a for a in found_assets if "DCCON" in a]
            
            # Case A: BESS Assets
            if bess_assets and not (inv_assets or mfm_assets or pcs_assets or pqm_assets or dccon_assets or weather_assets):
                thoughts.append(f"Analyzing BESS units: {', '.join(bess_assets)}")
                bess_results = {}
                for asset in bess_assets:
                    bess_results[asset] = ml_pipelines.get_bess_health(asset)
                context_data = f"[Diagnostics: BESS_Health={bess_results}]"
                if len(bess_results) > 1:
                    chart_spec = _chart_bess_comparison(bess_results)
                else:
                    chart_spec = _chart_bess(list(bess_results.values())[0])
            
            # Case B: Inverter Assets
            elif inv_assets and not (bess_assets or mfm_assets or pcs_assets or pqm_assets or dccon_assets or weather_assets):
                thoughts.append(f"Analyzing Inverter units: {', '.join(inv_assets)}")
                if len(inv_assets) == 1:
                    inv_id = inv_assets[0]
                    outlier_res = ml_pipelines.detect_scb_outliers(inv_id)
                    context_data = f"[Diagnostics: SCB={outlier_res}]"
                    chart_spec = _chart_scb(outlier_res)
                else:
                    eff_res = ml_pipelines.analyze_inverter_efficiency()
                    inv_compare = {}
                    label_val_pairs = []
                    for inv in inv_assets:
                        inv_key = inv.split("_")[-1]
                        short_key = inv_key.replace("JAMNAGAR", "").replace("VIRTUAL", "").replace("GATEWAY", "").replace("_", "")
                        inv_data = eff_res.get("per_inverter", {}).get(short_key, {"avg_efficiency_pct": 95.0, "status": "Normal"})
                        inv_compare[inv] = inv_data
                        label_val_pairs.append((short_key, inv_data.get("avg_efficiency_pct", 95.0)))
                    context_data = f"[Diagnostics: Inverter_Comparison={inv_compare}]"
                    chart_spec = _chart_comparison(label_val_pairs, "Inverter Conversion Efficiency Comparison (%)", "Efficiency %")

            # Case C: Weather Assets
            elif weather_assets and not (bess_assets or inv_assets or mfm_assets or pcs_assets or pqm_assets or dccon_assets):
                thoughts.append(f"Querying weather telemetry for: {', '.join(weather_assets)}")
                weather_res = get_generic_asset_data(weather_assets[0], limit=5)
                context_data = f"[Diagnostics: Weather_Data={weather_res}]"
                chart_spec = {
                    "type": "line",
                    "title": "Ambient vs Module Temp (°C)",
                    "data": {
                        "labels": [r["timestamp"][11:16] for r in reversed(weather_res)] if weather_res else [],
                        "datasets": [{
                            "label": "Ambient Temp",
                            "data": [r["ambientTemperature"] for r in reversed(weather_res)] if weather_res else [],
                            "borderColor": "#8ab833", "fill": False
                        }, {
                            "label": "Module Temp",
                            "data": [r["moduleTemperatureSensor01"] for r in reversed(weather_res)] if weather_res else [],
                            "borderColor": "#ef4444", "fill": False
                        }]
                    }
                }

            # Case D: Mixed or Generic telemetry assets (MFM, PCS, PQM, DCCON or mixed)
            else:
                target_assets = found_assets
                thoughts.append(f"Querying generic telemetry for: {', '.join(target_assets)}")
                results = {}
                for asset in target_assets:
                    results[asset] = get_generic_asset_data(asset, limit=5)
                context_data = f"[Diagnostics: Telemetry_Data={results}]"
                
                def determine_group_from_asset(asset_id):
                    if not asset_id:
                        return 'OTHER'
                    suffix = asset_id.split('_')[-1]
                    if 'INV' in suffix:
                        return 'INVERTER'
                    elif 'MFM' in suffix:
                        return 'METER'
                    elif 'BCT' in suffix:
                        return 'BESS'
                    elif 'PCS' in suffix:
                        return 'PCS'
                    elif 'WMS' in suffix or 'WEATHER' in suffix or 'WS' in suffix:
                        return 'WEATHER'
                    elif 'PQM' in suffix:
                        return 'PQM'
                    elif 'DCCON' in suffix:
                        return 'DCCON'
                    return 'OTHER'
                    
                def get_latest_value(record, grp):
                    if not record:
                        return 0.0
                    if grp == 'WEATHER':
                        return record.get('ambientTemperature') or record.get('globalHorizontalIrradiance') or 0.0
                    elif grp == 'BESS':
                        return record.get('bessSOC') or record.get('activePower') or 0.0
                    elif grp == 'INVERTER':
                        val = record.get('activePower')
                        if val is None and record.get('outputPower') is not None:
                            val = record.get('outputPower') / 1000.0
                        return val or 0.0
                    elif grp == 'METER':
                        return record.get('activePower') or record.get('voltageRPhase') or 0.0
                    elif grp == 'PCS':
                        return record.get('activePower') or 0.0
                    elif grp == 'PQM':
                        return record.get('activePower') or record.get('voltageRPhase') or 0.0
                    elif grp == 'DCCON':
                        return record.get('activePower') or record.get('inputVoltage') or 0.0
                    return record.get('activePower') or 0.0
                    
                def get_metric_label(grp):
                    if grp == 'WEATHER':
                        return "Ambient Temp / Irradiance"
                    elif grp == 'BESS':
                        return "Battery SOC (%)"
                    elif grp == 'INVERTER':
                        return "Active Power (kW)"
                    elif grp == 'METER':
                        return "Active Power (kW)"
                    elif grp == 'PCS':
                        return "Active Power (kW)"
                    elif grp == 'PQM':
                        return "Active Power (kW)"
                    elif grp == 'DCCON':
                        return "Active Power (kW)"
                    return "Value"

                label_val_pairs = []
                first_asset = target_assets[0]
                first_group = determine_group_from_asset(first_asset)
                metric_label = get_metric_label(first_group)
                
                for asset in target_assets:
                    records = results[asset]
                    short_name = asset.split("_")[-1]
                    grp = determine_group_from_asset(asset)
                    val = get_latest_value(records[0] if records else None, grp)
                    label_val_pairs.append((short_name, val))
                
                if len(target_assets) > 1:
                    chart_spec = _chart_comparison(label_val_pairs, f"Asset Telemetry Comparison — {metric_label}", metric_label)
                else:
                    asset = target_assets[0]
                    records = results[asset]
                    if records:
                        grp = determine_group_from_asset(asset)
                        field = "activePower"
                        for col in ["activePower", "bessSOC", "ambientTemperature", "voltageRPhase", "inputVoltage"]:
                            if col in records[0] and records[0][col] is not None:
                                field = col
                                break
                        chart_spec = {
                            "type": "line",
                            "title": f"{field} Trend — {asset.split('_')[-1]}",
                            "data": {
                                "labels": [r["timestamp"][11:16] for r in reversed(records)],
                                "datasets": [{
                                    "label": field,
                                    "data": [r[field] for r in reversed(records)],
                                    "borderColor": "#4ab5c4",
                                    "backgroundColor": "rgba(74,181,196,0.12)",
                                    "fill": True, "tension": 0.4
                                }]
                            }
                        }

        # 1. PR Gap / generation
        elif any(k in prompt_lower for k in ["pr gap", "performance ratio", "expected gen",
                "generation performance", " gap ", "loss attribution", "soiling rate",
                "panel clean", "pr breakdown"]):
            thoughts.append("Running PR Gap & generation analysis...")
            gen_res     = ml_pipelines.get_expected_vs_actual_generation()
            soiling_res = ml_pipelines.calibrate_soiling_rate()
            ticket_desc = (
                f"Jamnagar plant PR gap: {gen_res['gap_kwh']:,} kWh (PR={gen_res['pr_actual']}). "
                f"Soiling accounts for {gen_res['attribution']['Soiling']:,} kWh at {soiling_res['avg_daily_soiling_rate_pct']}%/day. "
                "Schedule module cleaning cycle immediately."
            )
            ticket_payload = agent_setup.generate_actionable_task_payload(
                "Module Cleaning Cycle — High Soiling Losses", "Warning",
                "JAMNAGAR_SOLAR_FIELD", ticket_desc
            )
            context_data = (
                f"[Diagnostics: Expected={gen_res['expected_kwh']} kWh, Actual={gen_res['actual_kwh']} kWh, "
                f"Gap={gen_res['gap_kwh']} kWh, PR={gen_res['pr_actual']}, "
                f"Soiling={gen_res['attribution']['Soiling']} kWh, Shading={gen_res['attribution']['Shading']} kWh, "
                f"HW_loss={gen_res['attribution']['Hardware Inefficiency']} kWh, "
                f"Curtailment={gen_res['attribution']['Grid Curtailment']} kWh, "
                f"Daily soiling rate={soiling_res['avg_daily_soiling_rate_pct']}%/day, "
                f"Cleaning dates={soiling_res['inferred_cleaning_dates'][:5]}, "
                f"Ticket={ticket_payload}]"
            )
            chart_spec = _chart_pr_gap(gen_res)

        # 2. SCB / string outliers
        elif any(k in prompt_lower for k in ["outlier", "string", "scb", "underperform"]):
            thoughts.append("Running SCB String current Z-score analysis...")
            inv_id = ("JAMNAGAR_VIRTUAL_GATEWAY_B3INV1" if "b3inv1" in prompt_lower
                      else "JAMNAGAR_VIRTUAL_GATEWAY_B1INV1")
            outlier_res = ml_pipelines.detect_scb_outliers(inv_id)
            ticket_payload = ""
            if "error" not in outlier_res and outlier_res.get("underperforming_strings"):
                ticket_desc = (
                    f"Inverter {outlier_res['inverterId']}: {len(outlier_res['underperforming_strings'])} "
                    f"underperforming strings at {outlier_res['timestamp']}. Inspect SCB connections."
                )
                ticket_payload = agent_setup.generate_actionable_task_payload(
                    f"String Anomaly — {outlier_res['inverterId'].split('_')[-1]}",
                    "Warning", outlier_res['inverterId'], ticket_desc
                )
            context_data = f"[Diagnostics: SCB={outlier_res}, Ticket={ticket_payload}]"
            chart_spec   = _chart_scb(outlier_res)

        # 3. BESS / battery
        elif any(k in prompt_lower for k in ["bess", "battery", "state of health", "coulombic", "cycle count"]):
            thoughts.append("Running BESS State-of-Health & Coulombic Efficiency analysis...")
            bess_res   = ml_pipelines.get_bess_health("JAMNAGAR_VIRTUAL_GATEWAY_B1BCT1")
            context_data = f"[Diagnostics: BESS={bess_res}]"
            chart_spec   = _chart_bess(bess_res)

        # 4. Inverter efficiency curve  ── ISING SPECIALIST ──
        elif any(k in prompt_lower for k in ["efficiency", "dc-ac", "dc ac", "load factor",
                "conversion", "inverter curve", "power curve"]):
            thoughts.append("Running Inverter DC-AC Efficiency Curve analysis (ML pipeline)...")
            eff_res = ml_pipelines.analyze_inverter_efficiency()
            ticket_payload = ""
            if eff_res.get("underperforming_inverters"):
                ticket_desc = (
                    f"Inverters {eff_res['underperforming_inverters']} below 92% DC-AC threshold. "
                    f"Fleet avg: {eff_res['fleet_avg_efficiency_pct']}%. Inspect power electronics."
                )
                ticket_payload = agent_setup.generate_actionable_task_payload(
                    "Inverter Efficiency Degradation — Inspection Required",
                    "Warning", "JAMNAGAR_INVERTER_FLEET", ticket_desc
                )
            context_data = (
                f"[Diagnostics: Fleet avg efficiency={eff_res['fleet_avg_efficiency_pct']}%, "
                f"Underperforming={eff_res['underperforming_inverters']}, "
                f"Per-inverter={eff_res['per_inverter']}, Curve={eff_res['fleet_curve']}, "
                f"Ticket={ticket_payload}]"
            )
            chart_spec = _chart_efficiency(eff_res)
            use_ising  = True

        # 5. Irradiance / POA correlation  ── ISING SPECIALIST ──
        elif any(k in prompt_lower for k in ["irradiance", "poa", "correlation", "clipping",
                "r2", "r²", "scatter", "output ratio"]):
            thoughts.append("Running Irradiance–Power Correlation analysis...")
            irr_res = ml_pipelines.analyze_irradiance_power_correlation()
            context_data = (
                f"[Diagnostics: R²={irr_res['correlation_r2']}, Clipping events={irr_res['clipping_events']}, "
                f"Output ratio={irr_res['avg_output_ratio']}, Flag={irr_res['anomaly_flag']}, "
                f"Sample data={irr_res['scatter_data'][:10]}]"
            )
            chart_spec = _chart_irradiance(irr_res)
            use_ising  = True

        # 6. Thermal anomaly detection  ── ISING SPECIALIST ──
        elif any(k in prompt_lower for k in ["thermal", "hotspot", "temperature", "noct",
                "heat", "temp delta", "module temp"]):
            thoughts.append("Running Module Thermal Anomaly Detection (NOCT model)...")
            thermal_res = ml_pipelines.detect_thermal_anomalies()
            ticket_payload = ""
            if thermal_res.get("anomaly_count", 0) > 0:
                ticket_desc = (
                    f"{thermal_res['anomaly_count']} hotspot events exceed {thermal_res['hotspot_threshold_c']}°C threshold. "
                    f"Max delta: {thermal_res['max_delta_c']}°C. Status: {thermal_res['status']}. "
                    "Dispatch thermal imaging crew."
                )
                ticket_payload = agent_setup.generate_actionable_task_payload(
                    f"Thermal Hotspot Alert — {thermal_res['anomaly_count']} Events",
                    "Warning", "JAMNAGAR_SOLAR_FIELD", ticket_desc
                )
            context_data = (
                f"[Diagnostics: Anomaly count={thermal_res['anomaly_count']}, "
                f"Avg delta={thermal_res['avg_delta_c']}°C, Max delta={thermal_res['max_delta_c']}°C, "
                f"Threshold={thermal_res['hotspot_threshold_c']}°C, Status={thermal_res['status']}, "
                f"Daily profile={thermal_res['daily_thermal_profile'][:7]}, Ticket={ticket_payload}]"
            )
            chart_spec = _chart_thermal(thermal_res)
            use_ising  = True

        # 7. Grid curtailment / availability  ── ISING SPECIALIST ──
        elif any(k in prompt_lower for k in ["curtailment", "availability", "curtail",
                "grid fault", "downtime", "operating state", "fault trip", "fault hour"]):
            thoughts.append("Running Grid Curtailment & Plant Availability analysis...")
            curt_res = ml_pipelines.analyze_grid_curtailment()
            ticket_payload = ""
            if curt_res.get("total_fault_hours", 0) > 2:
                ticket_desc = (
                    f"{curt_res['total_fault_hours']}h unplanned inverter trips. "
                    f"Plant availability: {curt_res['plant_availability_pct']}%. "
                    "Investigate trip root causes."
                )
                ticket_payload = agent_setup.generate_actionable_task_payload(
                    f"Inverter Fault Investigation — {curt_res['total_fault_hours']}h",
                    "Warning", "JAMNAGAR_VIRTUAL_GATEWAY", ticket_desc
                )
            context_data = (
                f"[Diagnostics: Availability={curt_res['plant_availability_pct']}%, "
                f"Curtailment={curt_res['total_curtailment_hours']}h, "
                f"Lost energy={curt_res['estimated_curtailed_kwh']} kWh, "
                f"Fault hours={curt_res['total_fault_hours']}, "
                f"Breakdown={curt_res['category_breakdown_hours']}, "
                f"Daily curtailment={curt_res['daily_curtailment'][:5]}, Ticket={ticket_payload}]"
            )
            chart_spec = _chart_curtailment(curt_res)
            use_ising  = True

        # ── Build LLM message ──────────────────────────────────────────────────
        system_instruction = (
            "You are the SunGenie AI Assistant (eAnalytiX Platform), an expert AI/ML solar engineering assistant.\n"
            "Your goal is to help operators monitor the Jamnagar Central Solar Plant using pre-computed telemetry & ML diagnostics.\n\n"
            "CRITICAL SYSTEM RULES:\n"
            "1. You are running in a text-only chat interface. You CANNOT execute Python code, call tools, run SQL queries, or invoke functions yourself. Any diagnostics are run in the background and injected into the user prompt if applicable.\n"
            "2. NEVER write mock tool calls, code blocks, or pretend to invoke functions (e.g., do NOT write `sql get_pr_gap_analysis(...)` or `json generate_actionable_task_payload(...)`).\n"
            "3. If no '[Diagnostics: ...]' context data is injected in the user prompt, and the query is NOT about your capabilities or how your calculations/models work, it means the requested asset or data does not exist in the Jamnagar database. Inform the operator that the requested asset is not found and suggest they query one of the active assets.\n"
            "4. Keep your responses direct, concise, and focused on final facts. No conversational filler or explanations of your internal reasoning process.\n"
            "5. If the user asks about your capabilities, what you can do, or the methodology/calculations of any of the 8 ML diagnostics pipelines, explain it directly using these engineering details:\n"
            "   - PR Gap Attribution: Expected solar power = Capacity * (POA/1000) * (1 - 0.004*(ModuleTemp - 25)) * LossFactor (LossFactor = 0.85, Capacity = 8648 kW). Energy in kWh is aggregated in 5-minute intervals. The gap (Expected - Actual) is attributed to Curtailment (Status = 2), Hardware efficiency drops, and Soiling/Shading using residual analysis.\n"
            "   - String SCB Outliers: Calculates Z-scores of String Combiner Box currents relative to the inverter average; current values with Z-score < -2.0 are flagged as outliers (indicating shading or faults).\n"
            "   - Soiling Loss Calibration: Identifies washer cleaning cycles by looking for positive daily PR jumps > 4%. The negative slope of daily PR decline between cleaning events represents the daily soiling loss rate.\n"
            "   - BESS Health & Cycles: Coulombic efficiency = sum(|discharge_current|) / sum(|charge_current|). Cycles are estimated by summing absolute SOC swings and dividing by 200%. State of Health (SoH) is calculated using linear capacity fade: 100% - (0.015% * total_cycles).\n"
            "   - Inverter DC-AC Efficiency: Divides outputPower by inputPVPower. Bins readings from 0-10% to 90-100% of the 1430 kW rated capacity per inverter. Flags underperforming units if efficiency is < 92%.\n"
            "   - Irradiance-Power Correlation: Computes R^2 linear correlation of POA vs active power. Flags power clipping when active power stays within 2% of rated capacity (8648 kW) despite POA exceeding 800 W/m^2.\n"
            "   - Module Thermal Hotspots: Uses standard NOCT predicted temp formula: T_predicted = T_ambient + ((NOCT - 20) / 800) * POA (with NOCT = 45C). Hotspot risk is flagged when measured temperature exceeds predicted by > 8C.\n"
            "   - Daylight Availability: Operating states (Running, standby, faults, maintenance) are classified only during daylight hours (POA > 50 W/m^2). Availability = Running Hours / Total Daylight Hours."
        )
        user_content = prompt
        if context_data:
            user_content = (
                f"{context_data}\n\n"
                f"You are the O&M AI assistant. The telemetry & ML diagnostics are pre-computed and "
                f"injected above. Answer this query directly using only that data: {prompt}\n\n"
                f"CRITICAL FORMATTING GUIDELINES:\n"
                f"1. Give ONLY the final numbers, conclusions, and facts. Absolutely no preamble or introductory transition.\n"
                f"2. DO NOT write or explain ANY formulas, equations, calculations, source code, SQL code, or intermediate pipeline methods in your response text. Treat all injected telemetry values as absolute final truth.\n"
                f"3. Summarize the key values and insights (e.g. State of Health, cycle count, efficiency percentages, temperatures, etc.) in a clear, bulleted list or short sentences.\n"
                f"4. If Diagnostics contains a non-empty Ticket, copy it verbatim inside a ```json``` block "
                f"under '### Actionable Task Ticket'."
            )

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": user_content})

        # ── Generic Nvidia caller ──────────────────────────────────────────────
        def _call_nvidia(model_name, api_key, temperature=0.5, max_tokens=4096):
            import urllib.request
            hdrs = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            body = json.dumps({
                "model":       model_name,
                "messages":    messages,
                "temperature": temperature,
                "max_tokens":  max_tokens,
            }).encode("utf-8")
            req = urllib.request.Request(NVIDIA_URL, data=body, headers=hdrs, method="POST")
            with urllib.request.urlopen(req, timeout=45) as resp:
                return resp.read().decode("utf-8")

        # ── Primary: Google Gemini (Gemma 4 26B) ──────────────────────────────────
        try:
            thoughts.append(f"Consulting Google Gemini ({MODEL_NAME})...")
            model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_instruction)
            response = await loop.run_in_executor(None, lambda: model.generate_content(user_content))
            return AgentResponse(response.text, thoughts, chart_spec)
        except Exception as e:
            thoughts.append(f"Gemini failed ({str(e)[:80]}). Falling back to Nvidia...")

        # ── Secondary fallback: Nvidia Ising Calibration 35B ──
        if use_ising:
            thoughts.append("Routing to Nvidia Ising Calibration 35B (physics specialist)...")
            try:
                raw  = await loop.run_in_executor(
                    None, lambda: _call_nvidia("nvidia/ising-calibration-1-35b-a3b",
                                               ISING_KEY, temperature=0.20, max_tokens=32768)
                )
                resp_json = json.loads(raw)
                choices   = resp_json.get("choices", [])
                if choices:
                    return AgentResponse(choices[0]["message"]["content"], thoughts, chart_spec)
                raise ValueError(f"Empty choices: {raw[:200]}")
            except Exception as ising_err:
                thoughts.append(f"Ising 35B unavailable ({str(ising_err)[:80]}). Falling back to Llama 3.1 8B...")
                use_ising = False

        # ── Tertiary fallback: Nvidia Llama 3.1 8B ──
        if not use_ising:
            thoughts.append("Consulting Nvidia Llama 3.1 8B (integrate.api.nvidia.com)...")
            try:
                raw  = await loop.run_in_executor(
                    None, lambda: _call_nvidia("meta/llama-3.1-8b-instruct",
                                               LLAMA_KEY, temperature=0.5, max_tokens=4096)
                )
                resp_json = json.loads(raw)
                choices   = resp_json.get("choices", [])
                if choices:
                    return AgentResponse(choices[0]["message"]["content"], thoughts, chart_spec)
                raise ValueError(f"Empty choices: {raw[:200]}")
            except Exception as llama_err:
                thoughts.append(f"Llama 8B failed ({str(llama_err)[:80]}).")
                return AgentResponse(f"All AI backends unavailable. Last error: {str(llama_err)}", thoughts)
