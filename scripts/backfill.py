"""Load past seasons into the database (run once, takes hours; safe to re-run).

    python -m scripts.backfill --league nba --seasons 2022 2023 2024 2025 2026
    python -m scripts.backfill --league nfl --seasons 2021 2022 2023 2024 2025

Season labels follow ESPN: NBA = season END year (2025-26 season is 2026),
NFL = season START year (2025 season, which ends Feb 2026, is 2025).

Resumable: every raw response is cached under data/raw/, so after a crash or
Ctrl+C just run the same command again; finished days are read from disk.
"""
import argparse
import calendar
import logging
import time
from datetime import date

from edge import db
from edge.config import league_config
from edge.http import DEFAULT_DELAY
from edge.ingest import refresh_details, refresh_schedule
from edge.sources.base import Game
from edge.timeutil import et_today


def season_days(league: str, season: int):
    today = et_today()
    for month, offset in league_config(league)["season_months"]:
        year = season + offset
        for d in range(1, calendar.monthrange(year, month)[1] + 1):
            day = date(year, month, d)
            if day < today:
                yield day


def main():
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True)
    ap.add_argument("--seasons", type=int, nargs="+", required=True)
    ap.add_argument("--no-details", action="store_true", help="schedule/results only (much faster)")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    args = ap.parse_args()

    import edge.http
    edge.http.DEFAULT_DELAY = args.delay
    db.init()

    done = set(db.read_sql(
        "SELECT DISTINCT o.game_key FROM odds_snapshots o WHERE o.kind = 'close'").game_key)

    for season in args.seasons:
        days = list(season_days(args.league, season))
        print(f"\n{args.league.upper()} {season}: {len(days)} days to scan")
        t0, total = time.time(), 0
        for i, day in enumerate(days, 1):
            try:
                games: list[Game] = refresh_schedule(args.league, day, use_cache=True)
            except Exception as e:  # noqa: BLE001
                print(f"  {day}: schedule failed ({e}); continuing")
                continue
            total += len(games)
            finished = [g for g in games if g.is_final and g.source_id and g.game_key not in done]
            if finished and not args.no_details:
                refresh_details(args.league, day, finished, use_cache=True, store_injuries=False)
            if i % 10 == 0 or i == len(days):
                mins = (time.time() - t0) / 60
                print(f"  {day}  {i}/{len(days)} days, {total} games, {mins:.0f} min elapsed")
    print("\nDone. Next: python -m scripts.train_model --league", args.league)


if __name__ == "__main__":
    main()
