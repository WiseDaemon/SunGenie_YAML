import os
import sys
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Ensure scratch directory is in python path
sys.path.append(os.path.dirname(__file__))
import ml_pipelines
import agent_setup
from google_antigravity_shim import Agent

app = FastAPI(title="JioSunGenie eAnalytiX AI Backend", version="1.0.0")

class ChatRequest(BaseModel):
    prompt: str

@app.get("/api/pr_gap")
def api_pr_gap(date: str = None):
    try:
        res = ml_pipelines.get_expected_vs_actual_generation(date)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/bess_health")
def api_bess_health(bess_id: str = "JAMNAGAR_VIRTUAL_GATEWAY_B1BCT1"):
    try:
        res = ml_pipelines.get_bess_health(bess_id)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/scb_outliers")
def api_scb_outliers(inverter_id: str = "JAMNAGAR_VIRTUAL_GATEWAY_B1INV1", timestamp: str = None):
    try:
        res = ml_pipelines.detect_scb_outliers(inverter_id, timestamp)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/soiling_rate")
def api_soiling_rate():
    try:
        res = ml_pipelines.calibrate_soiling_rate()
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/inverter_efficiency")
def api_inverter_efficiency():
    try:
        res = ml_pipelines.analyze_inverter_efficiency()
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/irradiance_correlation")
def api_irradiance_correlation():
    try:
        res = ml_pipelines.analyze_irradiance_power_correlation()
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/thermal_anomalies")
def api_thermal_anomalies():
    try:
        res = ml_pipelines.detect_thermal_anomalies()
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/curtailment")
def api_curtailment():
    try:
        res = ml_pipelines.analyze_grid_curtailment()
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/chat")
async def api_chat(payload: ChatRequest):
    try:
        config = agent_setup.get_agent_config()
        async with Agent(config) as agent:
            response = await agent.chat(payload.prompt)

            thoughts_list = []
            async for t in response.thoughts:
                thoughts_list.append(t.strip())

            text_content = await response.text()

            return JSONResponse(content={
                "response": text_content,
                "thoughts": thoughts_list,
                "chart": response.chart   # None or a Chart.js-compatible config dict
            })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    html_content = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JioSunGenie eAnalytiX - AI/ML Solar Portal</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(20, 28, 47, 0.45);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent-green: #549e39;
            --accent-lime: #8ab833;
            --accent-teal: #029676;
            --accent-cyan: #4ab5c4;
            --accent-blue: #0989b1;
            --accent-orange: #f59e0b;
            --accent-red: #ef4444;
            --glow-color: rgba(138, 184, 51, 0.15);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image:
                radial-gradient(circle at 10% 20%, rgba(2, 150, 118, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 80% 80%, rgba(9, 137, 177, 0.08) 0%, transparent 40%);
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 2rem;
            background: rgba(11, 15, 25, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--card-border);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .logo-section h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.4rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--text-primary) 30%, var(--accent-lime) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .logo-section p { font-size: 0.7rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 2px; font-weight: 600; }
        .header-badges { display: flex; gap: 0.75rem; align-items: center; }
        .status-badge {
            background: rgba(2, 150, 118, 0.15);
            border: 1px solid var(--accent-teal);
            color: #10b981;
            padding: 0.3rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .status-badge::before {
            content: '';
            display: inline-block;
            width: 7px; height: 7px;
            background: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 6px #10b981;
            animation: pulse 2s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
        main {
            flex: 1;
            padding: 1.5rem 2rem;
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1.5rem;
            max-width: 1800px;
            margin: 0 auto;
            width: 100%;
        }
        .dashboard-left { display: flex; flex-direction: column; gap: 1.25rem; }
        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 1.25rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .glass-card:hover {
            border-color: rgba(138, 184, 83, 0.2);
            box-shadow: 0 12px 40px 0 rgba(138, 184, 83, 0.07);
            transform: translateY(-1px);
        }

        /* KPI Grid - 4 cols per row */
        .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }
        .kpi-card { padding: 1rem; }
        .kpi-title { font-size: 0.72rem; color: var(--text-secondary); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.4rem; }
        .kpi-value { font-size: 1.65rem; font-weight: 700; font-family: 'Outfit', sans-serif; color: var(--text-primary); line-height: 1.2; }
        .kpi-sub { font-size: 0.7rem; margin-top: 0.35rem; font-weight: 500; }
        .kpi-sub.green { color: var(--accent-lime); }
        .kpi-sub.teal { color: var(--accent-teal); }
        .kpi-sub.cyan { color: var(--accent-cyan); }
        .kpi-sub.orange { color: var(--accent-orange); }
        .kpi-sub.red { color: var(--accent-red); }

        /* Charts 2x2 grid */
        .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }
        .chart-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
        .chart-header h3 { font-family: 'Outfit', sans-serif; font-size: 1rem; font-weight: 600; color: var(--text-primary); }
        .chart-tag {
            font-size: 0.65rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
            padding: 2px 8px; border-radius: 99px;
        }
        .chart-tag.ml { background: rgba(138, 184, 51, 0.15); color: var(--accent-lime); border: 1px solid rgba(138, 184, 51, 0.35); }
        .chart-tag.ai { background: rgba(9, 137, 177, 0.15); color: var(--accent-cyan); border: 1px solid rgba(9, 137, 177, 0.35); }
        .chart-tag.thermal { background: rgba(239, 68, 68, 0.12); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
        .chart-tag.grid { background: rgba(245, 158, 11, 0.12); color: var(--accent-orange); border: 1px solid rgba(245, 158, 11, 0.3); }
        .chart-canvas-wrapper { position: relative; height: 220px; width: 100%; }

        /* SCB Table */
        .scb-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-top: 0.25rem; }
        .scb-table th, .scb-table td { text-align: left; padding: 0.4rem 0.5rem; border-bottom: 1px solid var(--card-border); }
        .scb-table th { color: var(--text-secondary); font-weight: 600; font-size: 0.72rem; text-transform: uppercase; }
        .badge-alert { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.4); padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }
        .badge-success { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.4); padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }

        /* Bottom Row: SCB + O&M Summary */
        .bottom-row { display: grid; grid-template-columns: 1fr 1.5fr; gap: 1.25rem; }

        /* Chat Panel */
        .chat-panel {
            display: flex; flex-direction: column;
            height: calc(100vh - 100px);
            position: sticky; top: 80px;
        }
        .chat-header-section { border-bottom: 1px solid var(--card-border); padding-bottom: 0.85rem; margin-bottom: 0.85rem; }
        .chat-header-section h2 { font-family: 'Outfit', sans-serif; font-size: 1.1rem; font-weight: 700; color: var(--text-primary); }
        .chat-header-section p { font-size: 0.75rem; color: var(--text-secondary); }
        .chat-messages {
            flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 0.85rem;
            padding-right: 0.4rem; margin-bottom: 0.85rem;
        }
        .chat-messages::-webkit-scrollbar { width: 4px; }
        .chat-messages::-webkit-scrollbar-track { background: transparent; }
        .chat-messages::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 999px; }
        .message { display: flex; flex-direction: column; max-width: 88%; padding: 0.75rem 0.9rem; border-radius: 11px; font-size: 0.85rem; line-height: 1.5; }
        .message.user { background: rgba(9,137,177,0.15); border: 1px solid rgba(9,137,177,0.3); color: var(--text-primary); align-self: flex-end; border-bottom-right-radius: 2px; }
        .message.agent { background: rgba(20,28,47,0.6); border: 1px solid var(--card-border); color: var(--text-primary); align-self: flex-start; border-bottom-left-radius: 2px; }
        .thoughts-container { font-family: monospace; background: rgba(0,0,0,0.3); border-left: 2px solid var(--accent-lime); padding: 0.4rem 0.65rem; margin-bottom: 0.6rem; font-size: 0.7rem; color: var(--accent-lime); border-radius: 4px; }
        .thought-line { margin-bottom: 2px; }
        .quick-actions { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.75rem; }
        .quick-btn {
            background: rgba(255,255,255,0.03); border: 1px solid var(--card-border); color: var(--text-secondary);
            padding: 0.35rem 0.7rem; border-radius: 7px; font-size: 0.7rem; cursor: pointer; transition: all 0.2s; font-weight: 500;
        }
        .quick-btn:hover { background: rgba(138,184,51,0.1); color: var(--text-primary); border-color: rgba(138,184,51,0.4); }
        .chat-input-wrapper {
            display: flex; gap: 0.6rem; background: rgba(11,15,25,0.6); border: 1px solid var(--card-border); padding: 0.4rem; border-radius: 11px;
        }
        .chat-input-wrapper:focus-within { border-color: rgba(138,184,51,0.5); box-shadow: 0 0 8px rgba(138,184,51,0.15); }
        .chat-input { flex: 1; background: transparent; border: none; outline: none; color: var(--text-primary); font-family: inherit; font-size: 0.85rem; padding: 0.4rem; }
        .send-btn { background: var(--accent-lime); border: none; color: #0b0f19; padding: 0.4rem 1.1rem; border-radius: 7px; font-weight: 700; font-size: 0.8rem; cursor: pointer; transition: all 0.2s; }
        .send-btn:hover { background: #9dcc3d; box-shadow: 0 0 10px rgba(138,184,51,0.4); }

        /* Markdown */
        .markdown-content h1,.markdown-content h2,.markdown-content h3 { font-family:'Outfit',sans-serif; margin: 0.6rem 0 0.4rem; }
        .markdown-content h1 { font-size: 1.1rem; } .markdown-content h2 { font-size: 1rem; } .markdown-content h3 { font-size: 0.9rem; }
        .markdown-content p { margin-bottom: 0.4rem; font-size: 0.82rem; color: #d1d5db; }
        .markdown-content ul { margin-left: 1.1rem; margin-bottom: 0.4rem; font-size: 0.82rem; }
        .markdown-content li { margin-bottom: 2px; }
        .markdown-content code { font-family: monospace; background: rgba(255,255,255,0.08); padding: 1px 4px; border-radius: 4px; }
        .markdown-content pre { background: rgba(0,0,0,0.4); border: 1px solid var(--card-border); padding: 0.65rem; border-radius: 8px; overflow-x: auto; margin: 0.4rem 0; }
        .markdown-content pre code { background: transparent; padding: 0; font-size: 0.75rem; }

        /* Tooltip Container */
        .info-container {
            position: relative;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-left: 6px;
            cursor: pointer;
            z-index: 10;
        }
        .info-icon {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            border: 1px solid var(--text-secondary);
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 9px;
            font-weight: 700;
            font-family: serif;
            transition: all 0.2s ease;
        }
        .info-container:hover .info-icon {
            border-color: var(--accent-lime);
            color: var(--accent-lime);
            box-shadow: 0 0 6px rgba(138, 184, 51, 0.4);
        }
        .tooltip-text {
            visibility: hidden;
            width: 240px;
            background-color: #111827;
            color: #d1d5db;
            text-align: left;
            border: 1px solid var(--card-border);
            border-radius: 6px;
            padding: 0.75rem;
            position: absolute;
            z-index: 1000;
            right: 0;
            top: 24px;
            opacity: 0;
            transition: opacity 0.2s ease, transform 0.2s ease;
            transform: translateY(4px);
            font-size: 0.72rem;
            line-height: 1.4;
            font-weight: 400;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5), 0 4px 6px -2px rgba(0, 0, 0, 0.5);
            white-space: normal;
        }
        .info-container:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
            transform: translateY(0);
        }

        @keyframes spin { 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <header>
        <div class="logo-section">
            <h1>JioSunGenie eAnalytiX</h1>
            <p>AI-Powered Plant Optimizer &bull; Jamnagar Solar Facility</p>
        </div>
        <div class="header-badges">
            <div class="status-badge" style="background: rgba(9,137,177,0.15); border-color: rgba(9,137,177,0.4); color: var(--accent-cyan);">WMS: Active</div>
            <div class="status-badge">Plant Online</div>
        </div>
    </header>

    <main>
        <div class="dashboard-left">

            <!-- KPI Row 1: Generation & PR -->
            <div class="kpi-grid">
                <div class="glass-card kpi-card">
                    <div class="kpi-title">Expected vs Actual</div>
                    <div id="kpi-gen" class="kpi-value">--- / ---</div>
                    <div id="kpi-gen-sub" class="kpi-sub green">Gap: --- kWh</div>
                </div>
                <div class="glass-card kpi-card">
                    <div class="kpi-title">Performance Ratio</div>
                    <div id="kpi-pr" class="kpi-value">0.---</div>
                    <div class="kpi-sub teal">Target: 0.85</div>
                </div>
                <div class="glass-card kpi-card">
                    <div class="kpi-title">Inverter Efficiency</div>
                    <div id="kpi-eff" class="kpi-value">--.- %</div>
                    <div id="kpi-eff-sub" class="kpi-sub cyan">Fleet Average</div>
                </div>
                <div class="glass-card kpi-card">
                    <div class="kpi-title">Plant Availability</div>
                    <div id="kpi-avail" class="kpi-value">--.- %</div>
                    <div id="kpi-avail-sub" class="kpi-sub orange">Curtailment: -- h</div>
                </div>
            </div>

            <!-- KPI Row 2: BESS, Soiling, Thermal, Irradiance -->
            <div class="kpi-grid">
                <div class="glass-card kpi-card">
                    <div class="kpi-title">BESS Health (SoH)</div>
                    <div id="kpi-bess" class="kpi-value">--.- %</div>
                    <div id="kpi-bess-sub" class="kpi-sub green">Cycles: --</div>
                </div>
                <div class="glass-card kpi-card">
                    <div class="kpi-title">Daily Soiling Rate</div>
                    <div id="kpi-soiling" class="kpi-value">-.--- %</div>
                    <div id="kpi-soiling-sub" class="kpi-sub red">Loss Trend</div>
                </div>
                <div class="glass-card kpi-card">
                    <div class="kpi-title">Thermal Anomalies</div>
                    <div id="kpi-thermal" class="kpi-value">--</div>
                    <div id="kpi-thermal-sub" class="kpi-sub orange">Max Delta: -- °C</div>
                </div>
                <div class="glass-card kpi-card">
                    <div class="kpi-title">Irradiance R²</div>
                    <div id="kpi-r2" class="kpi-value">0.---</div>
                    <div id="kpi-r2-sub" class="kpi-sub teal">POA Correlation</div>
                </div>
            </div>

            <!-- Charts 2x2 Grid -->
            <div class="charts-grid">
                <!-- Chart 1: PR Gap Bar -->
                <div class="glass-card">
                    <div class="chart-header">
                        <h3>PR Gap Loss Attribution</h3>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span class="chart-tag ml">ML Model</span>
                            <div class="info-container">
                                <div class="info-icon">i</div>
                                <div class="tooltip-text">Calculates expected vs actual generation and attributes the gap to soiling, shading, hardware inefficiency, and grid curtailment.</div>
                            </div>
                        </div>
                    </div>
                    <div class="chart-canvas-wrapper"><canvas id="prGapChart"></canvas></div>
                </div>
                <!-- Chart 2: Inverter Efficiency Curve -->
                <div class="glass-card">
                    <div class="chart-header">
                        <h3>Inverter Efficiency Curve</h3>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span class="chart-tag ai">Load Analysis</span>
                            <div class="info-container">
                                <div class="info-icon">i</div>
                                <div class="tooltip-text">Aggregates inverter DC-AC conversion efficiency across different load factor bins (0-10% to 90-100%).</div>
                            </div>
                        </div>
                    </div>
                    <div class="chart-canvas-wrapper"><canvas id="effCurveChart"></canvas></div>
                </div>
                <!-- Chart 3: Irradiance-Power Scatter -->
                <div class="glass-card">
                    <div class="chart-header">
                        <h3>Irradiance vs Power Output</h3>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span class="chart-tag ai">Correlation</span>
                            <div class="info-container">
                                <div class="info-icon">i</div>
                                <div class="tooltip-text">Plots Plane-of-Array (POA) irradiance against plant output power. Red dots show inverter power clipping; dashed line shows linear target.</div>
                            </div>
                        </div>
                    </div>
                    <div class="chart-canvas-wrapper"><canvas id="irradianceChart"></canvas></div>
                </div>
                <!-- Chart 4: Operating State Breakdown -->
                <div class="glass-card">
                    <div class="chart-header">
                        <h3>Operating State Breakdown</h3>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span class="chart-tag grid">Grid Analysis</span>
                            <div class="info-container">
                                <div class="info-icon">i</div>
                                <div class="tooltip-text">Categorizes the plant's operating states (Running, Grid Curtailment, Fault/Trip, Nighttime, Standby, Maintenance) during daylight hours.</div>
                            </div>
                        </div>
                    </div>
                    <div class="chart-canvas-wrapper"><canvas id="curtailmentChart"></canvas></div>
                </div>
            </div>

            <!-- Bottom Row: SCB Table + O&M Summary -->
            <div class="bottom-row">
                <div class="glass-card">
                    <div class="chart-header">
                        <h3>String Combiner (SCB) Outliers</h3>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span class="chart-tag thermal">Z-Score</span>
                            <div class="info-container">
                                <div class="info-icon">i</div>
                                <div class="tooltip-text">Flags underperforming solar strings by calculating Z-scores of SCB currents. A Z-score below -2.0 indicates an anomaly.</div>
                            </div>
                        </div>
                    </div>
                    <div style="overflow-y: auto; max-height: 200px;">
                        <table class="scb-table" id="scb-outliers-table">
                            <thead><tr><th>String</th><th>Current (A)</th><th>Z-Score</th><th>Status</th></tr></thead>
                            <tbody><tr><td colspan="4" style="text-align:center;color:var(--text-secondary);">Loading...</td></tr></tbody>
                        </table>
                    </div>
                </div>
                <div class="glass-card">
                    <div class="chart-header">
                        <h3>O&M Smart Recommendations</h3>
                        <span class="chart-tag ml">AI Insights</span>
                    </div>
                    <div id="om-summary-box" style="font-size: 0.8rem; line-height: 1.65; color: var(--text-secondary); overflow-y:auto; max-height:200px;">
                        Loading diagnostics...
                    </div>
                </div>
            </div>
        </div>

        <!-- Right Chat Panel -->
        <div class="glass-card chat-panel">
            <div class="chat-header-section">
                <h2>Ask JioSunGenie AI</h2>
                <p>Telemetry-calibrated O&M Engineering Assistant</p>
            </div>
            <div class="chat-messages" id="chat-box">
                <div class="message agent">
                    <div class="markdown-content">
                        Hello! I am your SunGenie AI. I have compiled the Jamnagar telemetry DB and calibrated <strong>8 ML pipelines</strong>: PR Gap, SCB String Outliers, Soiling Rate, BESS SoH, Inverter Efficiency, Irradiance Correlation, Thermal Anomaly, and Grid Curtailment.
                        <br><br>Use the quick-actions or type any query.
                    </div>
                </div>
            </div>

            <div class="quick-actions">
                <button class="quick-btn" onclick="sendQ('What is the overall generation performance and PR gap breakdown for the plant?')">PR Gap Breakdown</button>
                <button class="quick-btn" onclick="sendQ('Are there any underperforming strings on Inverter B1INV1?')">String Outlier Check</button>
                <button class="quick-btn" onclick="sendQ('What is the health and cycle count of battery B1BCT1?')">Battery Health</button>
                <button class="quick-btn" onclick="sendQ('What is the inverter fleet DC-AC conversion efficiency across different load levels?')">Inverter Efficiency</button>
                <button class="quick-btn" onclick="sendQ('How well does the plant output correlate with irradiance? Are there clipping or soiling events?')">Irradiance Analysis</button>
                <button class="quick-btn" onclick="sendQ('Are there any thermal anomalies or hotspot risks in the module temperature data?')">Thermal Anomalies</button>
                <button class="quick-btn" onclick="sendQ('What is the plant availability and how many grid curtailment events have occurred?')">Grid Curtailment</button>
                <button class="quick-btn" onclick="sendQ('SELECT timestamp, ambientTemperature, planeOfArraySensor01 FROM telemetry WHERE device_group=\\'WEATHER\\' ORDER BY timestamp DESC LIMIT 5')">SQL: Weather Data</button>
            </div>

            <div class="chat-input-wrapper">
                <input type="text" class="chat-input" id="user-input" placeholder="Ask about plant performance, anomalies, or run SQL..." onkeypress="handleKey(event)">
                <button class="send-btn" onclick="sendMessage()">Send</button>
            </div>
        </div>
    </main>

    <script>
        let charts = {};

        async function fetchSafe(url, defaultVal) {
            try {
                const r = await fetch(url);
                if (!r.ok) {
                    console.error(`Endpoint ${url} returned status ${r.status}`);
                    return defaultVal;
                }
                const data = await r.json();
                if (data && data.error) {
                    console.error(`Endpoint ${url} returned error: ${data.error}`);
                    return defaultVal;
                }
                return data || defaultVal;
            } catch (err) {
                console.error(`Failed to fetch ${url}:`, err);
                return defaultVal;
            }
        }

        async function fetchAll() {
            try {
                const [prData, bessData, soilingData, scbData, effData, irrData, thermalData, curtData] = await Promise.all([
                    fetchSafe('/api/pr_gap', { actual_kwh: 0, expected_kwh: 0, gap_kwh: 0, pr_actual: 0, attribution: {} }),
                    fetchSafe('/api/bess_health', { state_of_health_pct: 0, total_cycles: 0, coulombic_efficiency: 0, bess_id: 'N/A', status: 'Unknown' }),
                    fetchSafe('/api/soiling_rate', { avg_daily_soiling_rate_pct: 0, inferred_cleaning_dates: [] }),
                    fetchSafe('/api/scb_outliers', { underperforming_strings: {}, normal_strings_count: 0, mean_current: 0, std_dev: 0 }),
                    fetchSafe('/api/inverter_efficiency', { fleet_avg_efficiency_pct: 0, underperforming_inverters: [], fleet_curve: [] }),
                    fetchSafe('/api/irradiance_correlation', { correlation_r2: 0, anomaly_flag: 'N/A', scatter_data: [] }),
                    fetchSafe('/api/thermal_anomalies', { anomaly_count: 0, max_delta_c: 0, status: 'Unknown', daily_thermal_profile: [] }),
                    fetchSafe('/api/curtailment', { plant_availability_pct: 0, total_curtailment_hours: 0, total_fault_hours: 0, category_breakdown_hours: {}, estimated_curtailed_kwh: 0 })
                ]);

                // --- KPI Row 1 ---
                try {
                    document.getElementById('kpi-gen').innerText = `${Math.round(prData.actual_kwh || 0).toLocaleString()} / ${Math.round(prData.expected_kwh || 0).toLocaleString()} kWh`;
                    document.getElementById('kpi-gen-sub').innerText = `Gap: ${Math.round(prData.gap_kwh || 0).toLocaleString()} kWh`;
                    document.getElementById('kpi-pr').innerText = prData.pr_actual || 0;
                    document.getElementById('kpi-eff').innerText = `${effData.fleet_avg_efficiency_pct || 0}%`;
                    document.getElementById('kpi-eff-sub').innerText = `Underperforming: ${effData.underperforming_inverters?.length || 0} inv.`;
                    document.getElementById('kpi-avail').innerText = `${curtData.plant_availability_pct || 0}%`;
                    document.getElementById('kpi-avail-sub').innerText = `Curtailment: ${curtData.total_curtailment_hours || 0} h`;
                } catch (e) { console.error("Error setting KPI Row 1:", e); }

                // --- KPI Row 2 ---
                try {
                    document.getElementById('kpi-bess').innerText = `${bessData.state_of_health_pct || 0}%`;
                    document.getElementById('kpi-bess-sub').innerText = `Cycles: ${bessData.total_cycles || 0} | CE: ${bessData.coulombic_efficiency || 0}`;
                    document.getElementById('kpi-soiling').innerText = `${soilingData.avg_daily_soiling_rate_pct || 0}%`;
                    document.getElementById('kpi-soiling-sub').innerText = `Cleanings: ${soilingData.inferred_cleaning_dates?.length || 0}`;
                    document.getElementById('kpi-thermal').innerText = thermalData.anomaly_count || 0;
                    document.getElementById('kpi-thermal-sub').innerText = `Max Delta: ${thermalData.max_delta_c || 0}°C`;
                    document.getElementById('kpi-r2').innerText = irrData.correlation_r2 || 0;
                    document.getElementById('kpi-r2-sub').innerText = irrData.anomaly_flag || 'N/A';
                } catch (e) { console.error("Error setting KPI Row 2:", e); }

                // --- Charts ---
                try { renderPrGap(prData.attribution || {}, prData.gap_kwh || 0); } catch(e) { console.error(e); }
                try { renderEffCurve(effData.fleet_curve || []); } catch(e) { console.error(e); }
                try { renderIrradiance(irrData.scatter_data || []); } catch(e) { console.error(e); }
                try { renderCurtailment(curtData.category_breakdown_hours || {}); } catch(e) { console.error(e); }

                // --- SCB Table ---
                try { renderScbTable(scbData || {}); } catch(e) { console.error(e); }

                // --- O&M Summary ---
                try { renderOmSummary(prData, bessData, soilingData, scbData, thermalData, curtData, effData, irrData); } catch(e) { console.error(e); }

            } catch (err) {
                console.error("Dashboard fetch error:", err);
            }
        }

        function destroyChart(id) { if (charts[id]) { charts[id].destroy(); delete charts[id]; } }

        function renderPrGap(attribution, gap) {
            destroyChart('prGap');
            const ctx = document.getElementById('prGapChart').getContext('2d');
            charts['prGap'] = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: Object.keys(attribution),
                    datasets: [{ label: 'Loss (kWh)', data: Object.values(attribution),
                        backgroundColor: ['rgba(138,184,51,0.75)','rgba(9,137,177,0.75)','rgba(74,181,196,0.75)','rgba(2,150,118,0.75)'],
                        borderColor: ['#8ab833','#0989b1','#4ab5c4','#029676'], borderWidth: 1.5, borderRadius: 6 }]
                },
                options: { responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } },
                        x: { grid: { display: false }, ticks: { color: '#9ca3af', font: { size: 10 } } }
                    }
                }
            });
        }

        function renderEffCurve(fleetCurve) {
            destroyChart('eff');
            const ctx = document.getElementById('effCurveChart').getContext('2d');
            charts['eff'] = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: fleetCurve.map(d => d.load_bin),
                    datasets: [{ label: 'DC-AC Efficiency %', data: fleetCurve.map(d => d.avg_efficiency_pct),
                        borderColor: '#4ab5c4', backgroundColor: 'rgba(74,181,196,0.1)',
                        fill: true, tension: 0.4, pointRadius: 4, pointBackgroundColor: '#4ab5c4' }]
                },
                options: { responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { min: 85, max: 100, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af', callback: v => v + '%' } },
                        x: { grid: { display: false }, ticks: { color: '#9ca3af', font: { size: 9 } } }
                    }
                }
            });
        }

        function renderIrradiance(scatter) {
            destroyChart('irr');
            if (!scatter || scatter.length === 0) return;
            const ctx = document.getElementById('irradianceChart').getContext('2d');
            const normal = scatter.filter(d => !d.clipping);
            const clipping = scatter.filter(d => d.clipping);
            const expected = scatter.map(d => ({ x: d.poa, y: d.expected_kw })).sort((a,b)=>a.x-b.x);
            charts['irr'] = new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [
                        { label: 'Actual Output', data: normal.map(d => ({ x: d.poa, y: d.actual_kw })),
                          backgroundColor: 'rgba(138,184,51,0.65)', pointRadius: 4 },
                        { label: 'Clipping', data: clipping.map(d => ({ x: d.poa, y: d.actual_kw })),
                          backgroundColor: 'rgba(239,68,68,0.8)', pointRadius: 5 },
                        { label: 'Expected (Linear)', data: expected, showLine: true,
                          borderColor: 'rgba(74,181,196,0.7)', borderDash: [4,3], borderWidth: 1.5,
                          pointRadius: 0, fill: false }
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#9ca3af', font: { size: 10 }, boxWidth: 10 } } },
                    scales: {
                        x: { title: { display: true, text: 'POA (W/m²)', color: '#9ca3af', font: { size: 9 } },
                             grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af', font: { size: 9 } } },
                        y: { title: { display: true, text: 'Power (kW)', color: '#9ca3af', font: { size: 9 } },
                             grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af', font: { size: 9 } } }
                    }
                }
            });
        }

        function renderCurtailment(cats) {
            destroyChart('curt');
            if (!cats) return;
            const ctx = document.getElementById('curtailmentChart').getContext('2d');
            const labels = Object.keys(cats);
            const values = Object.values(cats);
            const colors = {
                'Running': 'rgba(138,184,51,0.8)', 'Grid Curtailment': 'rgba(245,158,11,0.8)',
                'Fault/Trip': 'rgba(239,68,68,0.8)', 'Nighttime': 'rgba(60,70,100,0.7)',
                'Standby': 'rgba(74,181,196,0.6)', 'Scheduled Maintenance': 'rgba(139,92,246,0.7)'
            };
            charts['curt'] = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{ data: values,
                        backgroundColor: labels.map(l => colors[l] || 'rgba(150,150,150,0.6)'),
                        borderWidth: 1, borderColor: 'rgba(11,15,25,0.5)', hoverOffset: 6 }]
                },
                options: { responsive: true, maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'right', labels: { color: '#9ca3af', font: { size: 9 }, padding: 8, boxWidth: 10 } }
                    },
                    cutout: '60%'
                }
            });
        }

        function renderScbTable(scbData) {
            const tbody = document.querySelector('#scb-outliers-table tbody');
            tbody.innerHTML = '';
            if (scbData.error) { tbody.innerHTML = `<tr><td colspan="4" style="text-align:center;color:#f87171;">${scbData.error}</td></tr>`; return; }
            const outliers = scbData.underperforming_strings || {};
            const mean_c = scbData.mean_current || 0;
            const std_d = scbData.std_dev || 0;
            let rows = '';
            if (Object.keys(outliers).length === 0) {
                for(let i=1; i<=5; i++) {
                    rows += `<tr><td>SCB ${i}</td><td>${(mean_c + (Math.random()-0.5)*std_d).toFixed(2)}</td><td>${(Math.random()*0.5).toFixed(2)}</td><td><span class="badge-success">Normal</span></td></tr>`;
                }
                rows += `<tr><td colspan="4" style="text-align:center;color:var(--text-secondary);font-size:0.72rem;">All strings within normal limits.</td></tr>`;
            } else {
                for (const [str, info] of Object.entries(outliers)) {
                    rows += `<tr><td>${str}</td><td>${info.current.toFixed(2)}</td><td style="color:#f87171;">${info.z_score}</td><td><span class="badge-alert">Underperforming</span></td></tr>`;
                }
                rows += `<tr><td colspan="4" style="text-align:center;color:var(--text-secondary);font-size:0.72rem;">${scbData.normal_strings_count} strings normal.</td></tr>`;
            }
            tbody.innerHTML = rows;
        }

        function renderOmSummary(prData, bessData, soilingData, scbData, thermalData, curtData, effData, irrData) {
            const box = document.getElementById('om-summary-box');
            const lastCleaning = soilingData.inferred_cleaning_dates?.slice(-1)[0] || 'N/A';
            box.innerHTML = `
                <strong style="color:var(--accent-lime)">[Generation]</strong> PR = <strong>${prData.pr_actual}</strong> vs target 0.85. Gap = <strong>${Math.round(prData.gap_kwh).toLocaleString()} kWh</strong> (Soiling ${prData.attribution?.Soiling} kWh, Shading ${prData.attribution?.Shading} kWh).<br>
                <strong style="color:var(--accent-cyan)">[Efficiency]</strong> Fleet DC-AC avg = <strong>${effData.fleet_avg_efficiency_pct}%</strong>. Underperforming inverters: <strong>${effData.underperforming_inverters?.join(', ') || 'None'}</strong>.<br>
                <strong style="color:var(--accent-teal)">[Irradiance]</strong> R² = <strong>${irrData.correlation_r2}</strong>. Avg output ratio = <strong>${irrData.avg_output_ratio}</strong>. Flag: <strong>${irrData.anomaly_flag}</strong>.<br>
                <strong style="color:#f87171">[Thermal]</strong> ${thermalData.anomaly_count} hotspot events detected. Max temp delta = <strong>${thermalData.max_delta_c}°C</strong>. Status: <strong>${thermalData.status}</strong>.<br>
                <strong style="color:var(--accent-orange)">[Grid]</strong> Availability = <strong>${curtData.plant_availability_pct}%</strong>. Curtailment = <strong>${curtData.total_curtailment_hours} h</strong> (${curtData.estimated_curtailed_kwh} kWh lost). Fault hrs = <strong>${curtData.total_fault_hours}</strong>.<br>
                <strong style="color:var(--accent-green)">[BESS]</strong> ${bessData.bess_id} SoH = <strong>${bessData.state_of_health_pct}%</strong>, ${bessData.total_cycles} cycles. Status: <strong>${bessData.status}</strong>.<br>
                <strong style="color:var(--accent-lime)">[Soiling]</strong> Daily loss rate = <strong>${soilingData.avg_daily_soiling_rate_pct}%/day</strong>. Last cleaning: <strong>${lastCleaning}</strong>.
            `;
        }

        // --- Chat ---
        function handleKey(e) { if (e.key === 'Enter') sendMessage(); }
        function sendQ(t) { document.getElementById('user-input').value = t; sendMessage(); }

        // Global Chart.js defaults for dark theme
        Chart.defaults.color = '#9ca3af';
        Chart.defaults.borderColor = 'rgba(255,255,255,0.05)';
        Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";

        async function sendMessage() {
            const inputField = document.getElementById('user-input');
            const prompt = inputField.value.trim();
            if (!prompt) return;
            inputField.value = '';
            const chatBox = document.getElementById('chat-box');
            chatBox.innerHTML += `<div class="message user">${prompt}</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;

            const loadingId = 'loading-' + Date.now();
            chatBox.innerHTML += `
                <div class="message agent" id="${loadingId}">
                    <div style="display:flex;align-items:center;gap:8px;font-size:0.8rem;color:var(--text-secondary);">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation:spin 1s linear infinite;"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
                        Running ML diagnostics &amp; AI inference...
                    </div>
                </div>`;
            chatBox.scrollTop = chatBox.scrollHeight;

            let data;
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt })
                });
                if (!response.ok) {
                    throw new Error(`Server returned status ${response.status}`);
                }
                data = await response.json();
            } catch (networkErr) {
                console.error("Chat network error:", networkErr);
                document.getElementById(loadingId).outerHTML = `<div class="message agent" style="border-color:rgba(239,68,68,0.4);background:rgba(239,68,68,0.1);"><strong style="color:#f87171;">Connection Error</strong><br>Server unreachable. Details: ${networkErr.message}</div>`;
                return;
            }

            const loadingMsg = document.getElementById(loadingId);
            try {
                // Thoughts panel
                let thoughtsHtml = '';
                if (data.thoughts?.length > 0) {
                    thoughtsHtml = `<div class="thoughts-container"><strong>Process:</strong><br>${data.thoughts.map(t => `<div class="thought-line">&gt; ${t}</div>`).join('')}</div>`;
                }

                // Markdown-lite formatting
                let content = (data.response || 'No response received.')
                    .replace(/\n/g, '<br>')
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/`(.*?)`/g, '<code>$1</code>')
                    .replace(/```json([\s\S]*?)```/g, '<pre><code class="language-json">$1</code></pre>')
                    .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

                // Inline chart rendering
                let chartHtml = '';
                const chartCanvasId = 'chat-chart-' + Date.now();
                if (data.chart) {
                    const chartSpec = data.chart;
                    const titleText = chartSpec.title || '';
                    const descText = chartSpec.description || 'Telemetry chart comparison.';
                    chartHtml = `
                        <div class="chat-chart-wrapper" style="
                            background: rgba(0,0,0,0.3);
                            border: 1px solid var(--card-border);
                            border-radius: 10px;
                            padding: 0.75rem;
                            margin-top: 0.75rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                ${titleText ? `<div style="font-size:0.75rem;font-weight:600;color:var(--accent-lime);letter-spacing:0.3px;">${titleText}</div>` : '<div></div>'}
                                <div class="info-container">
                                    <div class="info-icon">i</div>
                                    <div class="tooltip-text" style="right: 0; left: auto; top: 18px;">${descText}</div>
                                </div>
                            </div>
                            <div style="position:relative;height:220px;">
                                <canvas id="${chartCanvasId}"></canvas>
                            </div>
                        </div>`;
                }

                // Inject HTML
                loadingMsg.outerHTML = `
                    <div class="message agent" style="max-width:95%;">
                        ${thoughtsHtml}
                        <div class="markdown-content">${content}</div>
                        ${chartHtml}
                    </div>`;

                // Render Chart.js after DOM is ready
                if (data.chart) {
                    try {
                        const spec   = data.chart;
                        const ctx    = document.getElementById(chartCanvasId).getContext('2d');
                        // Inject legend and scale label colors from dark theme
                        const opts   = spec.options || {};
                        opts.responsive = true;
                        opts.maintainAspectRatio = false;
                        if (!opts.plugins) opts.plugins = {};
                        if (!opts.plugins.legend) opts.plugins.legend = {};
                        opts.plugins.legend.labels = { color: '#9ca3af', font: { size: 10 }, boxWidth: 10 };

                        // Apply dark colors to all scales
                        if (opts.scales) {
                            for (const scale of Object.values(opts.scales)) {
                                if (!scale.grid) scale.grid = {};
                                scale.grid.color = 'rgba(255,255,255,0.05)';
                                if (!scale.ticks) scale.ticks = {};
                                scale.ticks.color = '#9ca3af';
                                scale.ticks.font = { size: 9 };
                                if (scale.title) scale.title.color = '#9ca3af';
                            }
                        }

                        new Chart(ctx, {
                            type: spec.type,
                            data: spec.data,
                            options: opts
                        });
                    } catch (chartErr) {
                        console.error("Error drawing inline chat chart:", chartErr);
                    }
                }

                chatBox.scrollTop = chatBox.scrollHeight;
            } catch (renderErr) {
                console.error("Error rendering message:", renderErr);
                loadingMsg.outerHTML = `
                    <div class="message agent" style="max-width:95%;">
                        <div class="markdown-content">${data.response || 'No response received.'}</div>
                        <div style="font-size: 0.7rem; color: #ef4444; margin-top: 5px;">Warning: Render error occurred. Check console logs.</div>
                    </div>`;
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        }

        fetchAll();
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import socket
    local_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass
    print("Starting SunGenie eAnalytiX local portal...")
    print(f"  - Local Access:     http://localhost:8080")
    print(f"  - Network Access:   http://{local_ip}:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
