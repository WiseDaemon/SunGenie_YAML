import os
import sys
import json
import secrets
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Ensure scratch directory is in python path
sys.path.append(os.path.dirname(__file__))
import ml_pipelines
import agent_setup
import config
import events_store
events_store.init_config_table()
from google_antigravity_shim import Agent

# --- Rate limiting (SEC-03) -------------------------------------------------
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="JioSunGenie SunGenie Platform AI Backend", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- API key auth (SEC-05) --------------------------------------------------
# Use the configured key, or generate an ephemeral one at startup so the
# server is never accidentally left open. The key is injected into the
# dashboard HTML at serve time, so the bundled UI keeps working either way.
API_KEY = config.API_KEY or secrets.token_urlsafe(24)
_API_KEY_IS_EPHEMERAL = config.API_KEY is None


@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    """Require a valid key on every /api/* route (header or query param)."""
    if request.url.path.startswith("/api/"):
        provided = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if not provided or not secrets.compare_digest(provided, API_KEY):
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized: missing or invalid API key."},
            )
    return await call_next(request)


class ChatRequest(BaseModel):
    prompt: str


class EventRequest(BaseModel):
    event_type: str
    asset_id: Optional[str] = None
    event_timestamp: Optional[str] = None
    value: Optional[float] = None
    notes: Optional[str] = None


class FeedbackRequest(BaseModel):
    prompt: str
    response: str
    rating: int
    label: Optional[str] = None


@app.post("/api/feedback")
@limiter.limit("60/minute")
def api_log_feedback(request: Request, payload: FeedbackRequest):
    try:
        rec = events_store.log_feedback(
            payload.prompt, payload.response, payload.rating, payload.label
        )
        return JSONResponse(content={"status": "logged", "feedback": rec})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/events")
@limiter.limit("60/minute")
def api_log_event(request: Request, payload: EventRequest):
    try:
        rec = events_store.log_event(
            payload.event_type, payload.asset_id, payload.event_timestamp,
            payload.value, payload.notes, source="dashboard",
        )
        return JSONResponse(content={"status": "logged", "event": rec})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/events")
@limiter.limit("60/minute")
def api_list_events(request: Request, event_type: str = None, asset_id: str = None, start_time: str = None, end_time: str = None):
    try:
        rows = events_store.list_events(event_type=event_type, asset_id=asset_id, start=start_time, end=end_time)
        return JSONResponse(content=rows)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/pr_gap")
@limiter.limit("60/minute")
def api_pr_gap(request: Request, date: str = None, start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.get_expected_vs_actual_generation(date_str=date, start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/bess_health")
@limiter.limit("60/minute")
def api_bess_health(request: Request, bess_id: str = "JAMNAGAR_VIRTUAL_GATEWAY_B1BCT1", start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.get_bess_health(bess_id, start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/scb_outliers")
@limiter.limit("60/minute")
def api_scb_outliers(request: Request, inverter_id: str = "JAMNAGAR_VIRTUAL_GATEWAY_B1INV1", timestamp: str = None, start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.detect_scb_outliers(inverter_id, timestamp_str=timestamp, start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/soiling_rate")
@limiter.limit("60/minute")
def api_soiling_rate(request: Request, start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.calibrate_soiling_rate(start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/inverter_efficiency")
@limiter.limit("60/minute")
def api_inverter_efficiency(request: Request, start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.analyze_inverter_efficiency(start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/irradiance_correlation")
@limiter.limit("60/minute")
def api_irradiance_correlation(request: Request, start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.analyze_irradiance_power_correlation(start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/thermal_anomalies")
@limiter.limit("60/minute")
def api_thermal_anomalies(request: Request, start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.detect_thermal_anomalies(start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/curtailment")
@limiter.limit("60/minute")
def api_curtailment(request: Request, start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.analyze_grid_curtailment(start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/alerts")
@limiter.limit("60/minute")
def api_alerts(request: Request, start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.get_all_alerts(start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/soiling_roi")
@limiter.limit("60/minute")
def api_soiling_roi(request: Request, start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.calibrate_cleaning_roi(start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/forecast")
@limiter.limit("60/minute")
def api_forecast(request: Request, start_time: str = None, end_time: str = None):
    try:
        res = ml_pipelines.get_generation_and_bess_forecast(start_time=start_time, end_time=end_time)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/data_range")
@limiter.limit("60/minute")
def api_data_range(request: Request):
    try:
        res = ml_pipelines.get_data_range()
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


class ConfigRequest(BaseModel):
    PLANT_CAPACITY_KW: Optional[float] = None
    LOSS_FACTOR: Optional[float] = None
    AVG_CURTAILED_KW: Optional[float] = None
    HARDWARE_LOSS_SHARE: Optional[float] = None


@app.get("/api/config")
@limiter.limit("60/minute")
def api_get_config(request: Request):
    try:
        res = events_store.get_config()
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/config")
@limiter.limit("30/minute")
def api_post_config(request: Request, payload: ConfigRequest):
    try:
        updates = {k: v for k, v in payload.dict().items() if v is not None}
        res = events_store.update_config(updates)
        return JSONResponse(content={"status": "updated", "config": res})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/generate_report")
@limiter.limit("30/minute")
def api_generate_report(request: Request, start_time: str = None, end_time: str = None, format: str = "csv"):
    try:
        from fastapi.responses import FileResponse
        file_path = ml_pipelines.generate_report(start_time=start_time, end_time=end_time, file_format=format)
        media_type = "text/csv" if format.lower() == "csv" else "text/html"
        filename = "sungenie_report.csv" if format.lower() == "csv" else "sungenie_report.html"
        return FileResponse(path=file_path, media_type=media_type, filename=filename)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/chat")
@limiter.limit("10/minute")
async def api_chat(request: Request, payload: ChatRequest):
    try:
        cfg = agent_setup.get_agent_config()
        async with Agent(cfg) as agent:
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
        # Inject the API key so the bundled dashboard can authenticate its /api/* calls.
        inject = f'<script>window.SUNGENIE_API_KEY = "{API_KEY}";</script>'
        if "window.SUNGENIE_API_KEY" in html_content:
            # Replace the placeholder declaration with the live key.
            import re
            html_content = re.sub(
                r'window\.SUNGENIE_API_KEY\s*=\s*"[^"]*";',
                f'window.SUNGENIE_API_KEY = "{API_KEY}";',
                html_content,
                count=1,
            )
        else:
            html_content = html_content.replace("</head>", inject + "\n</head>", 1)
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
    print("Starting SunGenie SunGenie Platform local portal...")
    print(f"  - Local Access:     http://localhost:8080")
    print(f"  - Network Access:   http://{local_ip}:8080")
    if _API_KEY_IS_EPHEMERAL:
        print("\n  [SEC-05] No SUNGENIE_API_KEY set in .env — generated an ephemeral key for this run:")
        print(f"           X-API-Key: {API_KEY}")
        print("           The bundled dashboard is auto-authenticated. Set SUNGENIE_API_KEY in .env for a stable key.\n")
    uvicorn.run(app, host="0.0.0.0", port=8080)
