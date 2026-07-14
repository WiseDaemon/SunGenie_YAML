import os
from dotenv import load_dotenv

def load_dotenv_resilient(path):
    if not os.path.exists(path):
        return
    content = None
    for enc in ('utf-8', 'utf-16', 'utf-16-le', 'utf-16-be'):
        try:
            with open(path, 'r', encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    
    if content is not None:
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception:
            pass
    load_dotenv(dotenv_path=path)

# Load environment variables.
env_path = os.environ.get("SUNGENIE_ENV_PATH")
if not env_path:
    windows_path = r"C:\LLM\.env"
    if os.path.exists(windows_path):
        env_path = windows_path
    else:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv_resilient(env_path)

# --- Database Configuration -------------------------------------------------
# BLOCK-02: default to the database that ships next to the code, so the project
# is portable. Override with SUNGENIE_DB_PATH for a custom location.
DB_PATH = os.environ.get(
    "SUNGENIE_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sungenie_telemetry.db"),
)

# Directory holding the raw event_history_report_*.csv files (used by compile_db.py).
DATA_DIR = os.environ.get("SUNGENIE_DATA_DIR", r"C:\LLM\SunGenie data")

# Separate DB for captured ground-truth events (cleanings, faults, capacity tests).
# Kept apart from the telemetry DB so re-running compile_db.py never wipes labels.
EVENTS_DB_PATH = os.environ.get(
    "SUNGENIE_EVENTS_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sungenie_events.db"),
)

# --- API Keys (read from environment / .env — never hardcode) ---------------
NVIDIA_ISING_KEY = os.environ.get("NVIDIA_ISING_KEY")
NVIDIA_LLAMA_KEY = os.environ.get("NVIDIA_LLAMA_KEY")
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# --- Server auth (SEC-05) ---------------------------------------------------
# Shared secret required on every /api/* request via the X-API-Key header
# (or ?api_key= query param for file downloads). If unset, app_server.py
# generates an ephemeral key at startup and prints it.
API_KEY = os.environ.get("SUNGENIE_API_KEY")
