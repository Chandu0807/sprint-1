"""config.py - settings come from environment variables (or a .env file next to this project)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mf_user:mf_pass@localhost:5432/microfinance")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")       # CHANGE in any real deployment
TOKEN_HOURS = int(os.getenv("TOKEN_HOURS", "8"))
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")]
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", ROOT / "ml" / "artifacts"))
