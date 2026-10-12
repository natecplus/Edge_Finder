"""Load the data behind the player matchup reports.

    python -m scripts.load_matchup_data --league nfl                 # last + current season
    python -m scripts.load_matchup_data --league nfl --seasons 2024 2025 2026
    python -m scripts.load_matchup_data --league nba                 # box scores from cached ESPN pages
    python -m scripts.load_matchup_data --league nba --fetch-missing # also download uncached games (slow)

NFL: downloads nflverse play-by-play, coverage charting and rosters (a few
     hundred MB for two seasons; a few minutes).
NBA: re-reads the ESPN game pages the backfill already saved, pulling each
     player's points, rebounds, assists, 3s and position. No new downloads
     unless --fetch-missing.
"""
import argparse
import json

from edge import db
from edge.config import RAW_DIR


def load_nfl(seasons: list[int]) -> None:
    from edge.sources import nflverse
    for s in seasons:
        for kind in nflverse.FILES:
            ok = nflverse.download(kind, s)
            db.log_source_run("nflverse", f"{kind}:{s}", ok=ok, rows=0,
                              error="" if ok else "not published yet")
            print(f"  {kind:<14} {s}: {'ok' if ok else 'not published yet'}")


def load_nba(fetch_missing: bool) -> None:
    from edge.sources.base import Game
    from edge.sources.espn import EspnDetail
    games = db.read_sql("SELECT * FROM games WHERE league = 'nba' AND source_id IS NOT NULL "
                        "AND home_score IS NOT NULL ORDER BY start_time")
    parser = EspnDetail(use_cache=True)
    done = missing = 0
    for r in games.itertuples(index=False):
        g = Game("nba", r.source_id, r.start_time, int(r.season), r.season_type, r.home_team, r.away_team,
                 int(r.home_score), int(r.away_score), r.status)
        cache = RAW_DIR / "espn" / "nba" / "summary" / f"{r.source_id}.json"
        if cache.exists():
            lines = parser.player_lines(g, json.loads(cache.read_text(encoding="utf-8")))
        elif fetch_missing:
            lines = parser.detail(g).lines
        else:
            missing += 1
            continue
        db.upsert_player_stats(lines)
        done += 1
        if done % 500 == 0:
            print(f"  {done} games ...")
    print(f"  box scores stored for {done} games; {missing} had no saved page"
          + (" (add --fetch-missing to download them)" if missing else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True, choices=["nfl", "nba"])
    ap.add_argument("--seasons", type=int, nargs="+")
    ap.add_argument("--fetch-missing", action="store_true")
    args = ap.parse_args()
    db.init()
    if args.league == "nfl":
        from edge.matchups_nfl import default_seasons
        load_nfl(args.seasons or default_seasons())
    else:
        load_nba(args.fetch_missing)
    print("Done. Open the Matchups page in the app.")


if __name__ == "__main__":
    main()
