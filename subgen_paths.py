import os
import sys
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parent


def default_data_dir():
    configured = os.environ.get("SUBGEN_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if getattr(sys, "frozen", False):
        local_app_data = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(local_app_data) / "SubGen"
    return SOURCE_DIR


DATA_DIR = default_data_dir()
CONFIG_PATH = DATA_DIR / "subgen_config.json"
ENV_PATH = DATA_DIR / ".env"
DB_PATH = DATA_DIR / "subgen.db"
UPLOADS_DIR = DATA_DIR / "uploads"
MODEL_CACHE_DIR = DATA_DIR / "models"
MOBILE_TOKEN_PATH = DATA_DIR / "mobile_access_token"
GOOGLE_DRIVE_CLIENT_PATH = DATA_DIR / "google_drive_client.json"
GOOGLE_DRIVE_TOKEN_PATH = DATA_DIR / "google_drive_token.json"
DRIVE_BATCHES_DIR = DATA_DIR / "drive_batches"


def ensure_data_directories():
    for path in (DATA_DIR, UPLOADS_DIR, MODEL_CACHE_DIR, DRIVE_BATCHES_DIR):
        path.mkdir(parents=True, exist_ok=True)


ensure_data_directories()
