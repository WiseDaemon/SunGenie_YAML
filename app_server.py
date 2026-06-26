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
def api_pr_gap(date: str = None, start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.get_expected_vs_actual_generation(date_str=date, start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/bess_health")
def api_bess_health(bess_id: str = "JAMNAGAR_VIRTUAL_GATEWAY_B1BCT1", start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.get_bess_health(bess_id, start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/scb_outliers")
def api_scb_outliers(inverter_id: str = "JAMNAGAR_VIRTUAL_GATEWAY_B1INV1", timestamp: str = None, start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.detect_scb_outliers(inverter_id, timestamp_str=timestamp, start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/soiling_rate")
def api_soiling_rate(start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.calibrate_soiling_rate(start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/inverter_efficiency")
def api_inverter_efficiency(start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.analyze_inverter_efficiency(start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/irradiance_correlation")
def api_irradiance_correlation(start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.analyze_irradiance_power_correlation(start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/thermal_anomalies")
def api_thermal_anomalies(start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.detect_thermal_anomalies(start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/curtailment")
def api_curtailment(start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.analyze_grid_curtailment(start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/alerts")
def api_alerts(start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.get_all_alerts(start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/soiling_roi")
def api_soiling_roi(start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.calibrate_cleaning_roi(start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/forecast")
def api_forecast(start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.get_generation_and_bess_forecast(start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/generate_report")
def api_generate_report(start_time: str = None, end_time: str = None, format: str = "csv"):
    try:
        from fastapi.responses import FileResponse
        file_path = ml_pipelines.generate_report(start_time=start_time, end_time=end_time, file_format=format)
        media_type = "text/csv" if format.lower() == "csv" else "text/html"
        filename = "sungenie_report.csv" if format.lower() == "csv" else "sungenie_report.html"
        return FileResponse(path=file_path, media_type=media_type, filename=filename)
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
    try:
        filepath = os.path.join(os.path.dirname(__file__), "index.html")
        with open(filepath, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        return HTMLResponse(content=f"<h3>Error loading index.html: {str(e)}</h3>", status_code=500)

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
