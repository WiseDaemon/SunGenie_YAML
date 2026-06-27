# Running SunGenie in Docker

Portable, reproducible deployment of the eAnalytiX backend + dashboard.

## Prerequisites
- Docker + Docker Compose
- `sungenie_telemetry.db` present in this folder (built via `python compile_db.py`)
- A `.env` file — copy the template and fill in your keys:
  ```
  cp .env.example .env
  ```
  Set at least `GEMINI_API_KEY`; set `SUNGENIE_API_KEY` to a long random string for a
  stable dashboard key (otherwise one is generated per run and printed to the logs).

## Build & run
```
docker compose up --build
```
Then open http://localhost:8080.

## Notes
- The telemetry DB is **mounted read-only**, not baked into the image, so the image stays small.
- Captured ground-truth events persist in `./events_data/` (and reports in `./reports/`) on the host.
- To run without Compose:
  ```
  docker build -t sungenie .
  docker run -p 8080:8080 --env-file .env \
    -v "$PWD/sungenie_telemetry.db:/app/sungenie_telemetry.db:ro" \
    -v "$PWD/events_data:/app/events_data" \
    sungenie
  ```
- The container runs `uvicorn app_server:app` on port 8080; the API-key auth and
  rate limiting apply exactly as in a local run.
