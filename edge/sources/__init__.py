"""Source registry + automatic failover.

Each kind of data has an ordered list of sources. `fetch_with_fallback` tries
them in order, logs every attempt to the source_runs table (that's what the
Data Health panel shows), and returns the first non-empty answer.
"""
from .base import Game, GameDetail, InjuryReport, OddsSnapshot, PlayerMinutes  # noqa: F401
from .bref import BrefSchedule
from .espn import EspnDetail, EspnSchedule
from .sbr import SbrOdds


def schedule_sources(use_cache: bool = False):
    return [EspnSchedule(use_cache), BrefSchedule(use_cache)]


def detail_sources(use_cache: bool = False):
    return [EspnDetail(use_cache)]


def odds_backup_sources(use_cache: bool = False):
    return [SbrOdds(use_cache)]


class AllSourcesFailed(RuntimeError):
    pass


def fetch_with_fallback(sources, method: str, *args, task: str = "", allow_empty: bool = False):
    """Call `method` on each source until one returns data.

    allow_empty=True means "an empty answer is a real answer" (e.g. no games
    today), so we stop at the first source that doesn't raise.
    """
    from edge.db import log_source_run   # local import avoids a circular import

    errors = []
    for src in sources:
        try:
            rows = getattr(src, method)(*args)
            n = len(rows) if hasattr(rows, "__len__") else 1
            log_source_run(src.name, task or method, ok=True, rows=n)
            if rows or allow_empty:
                return rows
        except Exception as e:  # noqa: BLE001 - we want to log any failure and move on
            log_source_run(src.name, task or method, ok=False, error=f"{type(e).__name__}: {e}")
            errors.append(f"{src.name}: {e}")
    if allow_empty and not errors:
        return []
    raise AllSourcesFailed("All sources failed: " + "; ".join(errors) if errors else "no data")
