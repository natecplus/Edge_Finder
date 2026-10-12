"""nflverse data files: free NFL play-by-play, coverage charting and rosters.

These are plain files published on GitHub (not an API), downloaded once and
kept in data/raw/nflverse/. The current season is re-downloaded weekly.

  play_by_play_{season}        every play: passer, receiver, rusher, yards, EPA
  pbp_participation_{season}   per play: man/zone, coverage type (Cover 1/2/3...),
                               route of the targeted receiver, pressure (FTN charting)
  roster_{season}              player positions

If a file isn't published yet (e.g. charting for the newest season), the
loader returns what exists and the matchup report says so.
"""
import io
import logging

import pandas as pd

from edge.config import RAW_DIR
from edge.http import get

log = logging.getLogger(__name__)
BASE = "https://github.com/nflverse/nflverse-data/releases/download"
DIR = RAW_DIR / "nflverse"

FILES = {
    "pbp": ("pbp", "play_by_play_{season}"),
    "participation": ("pbp_participation", "pbp_participation_{season}"),
    "roster": ("rosters", "roster_{season}"),
}

PBP_COLS = ["game_id", "play_id", "season", "week", "season_type", "posteam", "defteam",
            "play_type", "pass", "rush", "qb_dropback", "sack", "two_point_attempt",
            "passer_player_id", "passer_player_name", "receiver_player_id", "receiver_player_name",
            "rusher_player_id", "rusher_player_name", "yards_gained", "epa", "complete_pass", "air_yards"]
PART_COLS = ["nflverse_game_id", "play_id", "defense_man_zone_type", "defense_coverage_type",
             "route", "was_pressure"]
ROSTER_COLS = ["gsis_id", "full_name", "position", "team", "season"]


def path(kind: str, season: int):
    return DIR / f"{FILES[kind][1].format(season=season)}.parquet"


def download(kind: str, season: int) -> bool:
    """Fetch one file (parquet, falling back to csv). Returns False if not published."""
    tag, name = FILES[kind]
    name = name.format(season=season)
    DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("parquet", "csv.gz", "csv"):
        try:
            r = get(f"{BASE}/{tag}/{name}.{ext}", delay=1.0, timeout=120, retries=1)
        except Exception as e:  # noqa: BLE001 - 404 = not published yet
            log.info("nflverse %s.%s: %s", name, ext, e)
            continue
        if ext == "parquet":
            path(kind, season).write_bytes(r.content)
        else:
            df = pd.read_csv(io.BytesIO(r.content), compression="gzip" if ext == "csv.gz" else None,
                             low_memory=False)
            df.to_parquet(path(kind, season), index=False)
        return True
    return False


def _read(kind: str, seasons: list[int], cols: list[str]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        p = path(kind, s)
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        frames.append(df[[c for c in cols if c in df.columns]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def load(seasons: list[int]) -> dict:
    """{'pbp', 'participation', 'roster'} DataFrames for the given seasons (whatever is on disk)."""
    return {
        "pbp": _read("pbp", seasons, PBP_COLS),
        "participation": _read("participation", seasons, PART_COLS),
        "roster": _read("roster", seasons, ROSTER_COLS),
    }


def available(seasons: list[int]) -> dict:
    return {kind: [s for s in seasons if path(kind, s).exists()] for kind in FILES}
