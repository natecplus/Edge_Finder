"""Paths and league settings.

Everything that differs between leagues lives in config/leagues/<league>.yaml,
so the rest of the code never says "if league == 'nba'" for a number.
"""
import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config" / "leagues"

# Set EDGE_DATA_DIR to keep a separate database (the demo uses data/demo).
DATA_DIR = Path(os.environ.get("EDGE_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "edge.db"


def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)


@lru_cache
def league_config(league: str) -> dict:
    path = CONFIG_DIR / f"{league}.yaml"
    if not path.exists():
        raise ValueError(f"No config for league '{league}'. Expected {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def all_leagues() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
