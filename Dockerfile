# SunGenie eAnalytiX backend
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code (telemetry .db is excluded via .dockerignore and
# mounted at runtime — see docker-compose.yml — to keep the image small)
COPY . .

EXPOSE 8080

# Run the ASGI app directly (app_server.py also has a __main__ block for bare `python` runs)
CMD ["uvicorn", "app_server:app", "--host", "0.0.0.0", "--port", "8080"]
