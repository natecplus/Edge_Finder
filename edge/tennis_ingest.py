"""Getting tennis data into the database.

  load_history       tennis-data.co.uk results + closing odds (backfill, weekly refresh)
  refresh_tennis_day ESPN matches for a day (schedule, live results, any ESPN odds)
  refresh_odds_api   optional live odds, throttled to protect the free quota

The same match can arrive from two sources with slightly different names and
dates, so every write first looks for an existing row with the same two players
within 2 days and reuses its game_key instead of creating a duplicate.
"""
import logging
import re
from dataclasses import replace
from datetime import date, timedelta

import pandas as pd

from edge import db
from edge.config import league_config
from edge.players import PlayerIndex, ordered, player_key
from edge.sources import fetch_with_fallback
from edge.sources.espn_tennis import EspnTennis, TennisMatch
from edge.sources.odds_api import OddsApiTennis
from edge.sources.tennis_data import TennisDataHistory
from edge.timeutil import parse_utc, utcnow

log = logging.getLogger(__name__)
GENERIC = {"masters", "open", "atp", "championships", "championship", "the", "de", "rolex",
           "presented", "by", "international", "cup", "tour", "finals", "1000", "500", "250", "tennis"}


class Existing:
    """Index of stored matches by player pair, for cross-source de-duplication."""

    def __init__(self, league: str):
        g = db.read_sql("SELECT game_key, home_team, away_team, start_time, home_score FROM games "
                        "WHERE league = :lg", lg=league)
        self.by_pair = {}
        for r in g.itertuples(index=False):
            pair = tuple(sorted((player_key(r.home_team), player_key(r.away_team))))
            self.by_pair.setdefault(pair, []).append(
                (parse_utc(r.start_time), r.game_key, r.home_score is not None and r.home_score == r.home_score))
        self.names = set(g.home_team) | set(g.away_team)

    def find(self, home: str, away: str, start: str, days: int = 2, final: bool = True) -> str | None:
        """An upcoming match is never the same as an already-finished one."""
        pair = tuple(sorted((player_key(home), player_key(away))))
        t = parse_utc(start)
        for when, key, done in self.by_pair.get(pair, []):
            if done and not final:
                continue
            if abs((when - t).total_seconds()) <= days * 86400:
                return key
        return None

    def add(self, home, away, start, key, final: bool = True):
        pair = tuple(sorted((player_key(home), player_key(away))))
        self.by_pair.setdefault(pair, []).append((parse_utc(start), key, final))


def _rekey(m: TennisMatch, key: str) -> TennisMatch:
    m.meta["game_key"] = key
    m.odds = [replace(o, game_key=key) for o in m.odds]
    return m


def _write(league: str, matches: list[TennisMatch], prefer_existing_scores: bool = False) -> dict:
    """Insert new matches; attach meta/odds (and fresh results) to matches we already have."""
    existing = Existing(league)
    new_games, metas, odds, updated = [], [], [], 0
    for m in matches:
        g = m.game
        key = existing.find(g.home_team, g.away_team, g.start_time, final=g.is_final)
        if key is None:
            existing.add(g.home_team, g.away_team, g.start_time, g.game_key, g.is_final)
            new_games.append(g)
            key = g.game_key
        elif g.is_final and not prefer_existing_scores:
            with db.get_engine().begin() as conn:
                row = conn.execute(db.games.select().where(db.games.c.game_key == key)).mappings().first()
                if row and row["home_score"] is None:
                    # result arrived for a match we stored as upcoming; keep the stored player order
                    flip = player_key(row["home_team"]) != player_key(g.home_team)
                    hs, as_ = (g.away_score, g.home_score) if flip else (g.home_score, g.away_score)
                    conn.execute(db.games.update().where(db.games.c.game_key == key)
                                 .values(home_score=hs, away_score=as_, status=g.status))
                    updated += 1
        _rekey(m, key)
        metas.append({k: v for k, v in m.meta.items()})
        odds.extend(m.odds)
    db.upsert_games(new_games)
    db.upsert_tennis_meta(metas)
    db.insert_odds(odds)
    return {"new": len(new_games), "updated": updated, "odds": len(odds)}


# ------------------------------------------------------------------ history
def load_history(league: str, years: list[int], use_cache: bool = True) -> dict:
    src = TennisDataHistory(use_cache)
    totals = {}
    for y in years:
        try:
            matches = src.season(league, y)
            db.log_source_run(src.name, f"history:{league}", ok=True, rows=len(matches))
        except Exception as e:  # noqa: BLE001
            db.log_source_run(src.name, f"history:{league}", ok=False, error=f"{type(e).__name__}: {e}")
            log.warning("tennis-data %s failed: %s", y, e)
            continue
        totals[y] = _write(league, matches)
    return totals


# ------------------------------------------------------------------ surface
def _tokens(name: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+", (name or "").lower()) if t not in GENERIC}


def infer_surface(tournament: str, meta: pd.DataFrame) -> str:
    known = meta.dropna(subset=["surface"])
    known = known[known.surface.isin(["Hard", "Clay", "Grass", "Carpet"])]
    want = _tokens(tournament)
    best, score = "Hard", 0
    for t, s in known.groupby("tournament").surface.last().items():
        overlap = len(want & _tokens(t))
        if overlap > score:
            best, score = s, overlap
    return best


# ------------------------------------------------------------------ live
def refresh_tennis_day(league: str, day: date, use_cache: bool = False) -> dict:
    matches = fetch_with_fallback([EspnTennis(use_cache)], "matches_on", league, day,
                                  task=f"schedule:{league}", allow_empty=True)
    index = PlayerIndex(Existing(league).names)
    meta = db.tennis_meta_df(league)
    out = []
    for m in matches:
        g = m.game
        a, b = index.resolve(g.home_team), index.resolve(g.away_team)
        home, away = ordered(a, b)
        flip = home != a
        g2 = replace(g, home_team=home, away_team=away,
                     home_score=g.away_score if flip else g.home_score,
                     away_score=g.home_score if flip else g.away_score)
        odds = [replace(o, home_ml=o.away_ml, away_ml=o.home_ml) if flip else o for o in m.odds]
        m2 = TennisMatch(g2, {**m.meta, "surface": infer_surface(m.meta.get("tournament", ""), meta)}, odds)
        out.append(_rekey(m2, g2.game_key))
    result = _write(league, out)
    result["games"] = len(out)
    result.update(refresh_odds_api(league))
    return result


def refresh_odds_api(league: str, force: bool = False) -> dict:
    api = OddsApiTennis()
    if not api.enabled:
        return {"odds_api": "off (set ODDS_API_KEY to enable)"}
    hours = league_config(league).get("odds_api_min_hours", 4)
    last = db.read_sql("SELECT MAX(ran_at) AS t FROM source_runs WHERE source = 'odds-api' AND ok = 1 "
                       "AND task = :t", t=f"odds:{league}").t.iloc[0]
    if not force and isinstance(last, str) and utcnow() - parse_utc(last) < timedelta(hours=hours):
        return {"odds_api": f"skipped (last call under {hours}h ago)"}
    existing = Existing(league)
    index = PlayerIndex(existing.names)
    try:
        pairs = api.fetch(league, index.resolve)
        db.log_source_run(api.name, f"odds:{league}", ok=True, rows=len(pairs))
    except Exception as e:  # noqa: BLE001
        db.log_source_run(api.name, f"odds:{league}", ok=False, error=f"{type(e).__name__}: {e}")
        return {"odds_api": f"failed: {e}"}
    snaps = []
    for game, odds in pairs:
        key = existing.find(game.home_team, game.away_team, game.start_time, days=1, final=False)
        if key is None:
            continue           # only price matches ESPN has scheduled
        snaps.extend(replace(o, game_key=key) for o in odds)
    db.insert_odds(snaps)
    return {"odds_api": f"{len(snaps)} prices"}

