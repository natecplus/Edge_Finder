"""Pull data from the sources into the database.

These functions are what the scheduler calls. They only talk to sources
through fetch_with_fallback / the adapters, and only write through edge.db.
"""
import logging
from dataclasses import replace
from datetime import date

from edge import db
from edge.sources import (AllSourcesFailed, Game, detail_sources, fetch_with_fallback,
                          odds_backup_sources, schedule_sources)
from edge.timeutil import parse_utc, utcnow

log = logging.getLogger(__name__)


def refresh_schedule(league: str, day: date, use_cache: bool = False) -> list[Game]:
    games = fetch_with_fallback(schedule_sources(use_cache), "games_on", league, day,
                                task=f"schedule:{league}", allow_empty=True)
    db.upsert_games(games)
    return games


def refresh_details(league: str, day: date, games: list[Game], use_cache: bool = False,
                    store_injuries: bool = True) -> dict:
    """Odds + injuries for upcoming games, closing odds + minutes for finished ones."""
    counts = {"odds": 0, "injuries": 0, "minutes": 0, "missing_odds": []}
    now = utcnow()
    for game in games:
        started = parse_utc(game.start_time) <= now
        try:
            detail = fetch_with_fallback(detail_sources(use_cache), "detail", game,
                                         task=f"detail:{league}", allow_empty=True)
        except AllSourcesFailed as e:
            log.warning("detail failed for %s: %s", game.game_key, e)
            counts["missing_odds"].append(game.game_key)
            continue
        counts["odds"] += db.insert_odds(detail.odds)
        # Injury reports only count as information if captured BEFORE the game.
        # Storing a post-game report would leak the future into training data.
        if store_injuries and not started:
            counts["injuries"] += db.insert_injuries(detail.injuries)
        if game.is_final:
            counts["minutes"] += db.upsert_minutes(detail.minutes)
        if not detail.odds and not started:
            counts["missing_odds"].append(game.game_key)

    # Backup: if ESPN had no odds for some upcoming games, try SportsBookReview once for the day.
    if counts["missing_odds"]:
        try:
            backup = fetch_with_fallback(odds_backup_sources(use_cache), "odds_on", league, day,
                                         task=f"odds_backup:{league}")
            wanted = set(counts["missing_odds"])
            counts["odds"] += db.insert_odds([s for s in backup if s.game_key in wanted])
        except AllSourcesFailed as e:
            log.warning("odds backup failed: %s", e)
    return counts


def refresh_day(league: str, day: date, use_cache: bool = False, details: bool = True) -> dict:
    games = refresh_schedule(league, day, use_cache)
    out = {"games": len(games)}
    if details and games:
        # A backup schedule source has no ESPN id, but the row may already have one
        # from an earlier ESPN run, so look it up before giving up on details.
        known = db.read_sql("SELECT game_key, source_id FROM games WHERE league = :lg "
                            "AND source_id IS NOT NULL", lg=league)
        ids = dict(zip(known.game_key, known.source_id))
        games = [g if g.source_id else replace(g, source_id=ids.get(g.game_key)) for g in games]
        with_ids = [g for g in games if g.source_id]
        out.update(refresh_details(league, day, with_ids, use_cache))
    return out
