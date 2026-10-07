"""ESPN adapter: the primary source for schedules, odds, injuries and box scores.

These are the same public JSON endpoints ESPN's own site loads. They're free and
need no key, but they're undocumented, so every field access here is defensive
(`.get()` with defaults) and every format is pinned by a fixture test.
"""
from datetime import date

from edge.config import RAW_DIR, league_config
from edge.http import get_json
from edge.teams import normalize
from edge.timeutil import parse_utc, to_iso
from .base import (DetailSource, Game, GameDetail, InjuryReport, OddsSnapshot,
                   PlayerMinutes, ScheduleSource)

BASE = "https://site.api.espn.com/apis/site/v2/sports"
SEASON_TYPES = {1: "preseason", 2: "regular", 3: "postseason", 4: "offseason"}


def _score(team: dict, done: bool) -> int | None:
    if not done:
        return None
    try:
        return int(float(team.get("score")))
    except (TypeError, ValueError):
        return None


def parse_ml(value) -> int | None:
    """American odds come as -150, '-150', '+130', 'EVEN' or 'OFF'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value) if value != 0 else None
    s = str(value).strip().upper()
    if s in ("EVEN", "EV", "PK"):
        return 100
    try:
        n = int(float(s.replace("+", "")))
        return n if n != 0 else None
    except ValueError:
        return None


def _minutes(value) -> float | None:
    if value in (None, "", "--", "DNP"):
        return None
    s = str(value)
    try:
        if ":" in s:
            m, sec = s.split(":")[:2]
            return int(m) + int(sec) / 60
        return float(s)
    except ValueError:
        return None


class EspnSchedule(ScheduleSource):
    name = "espn"

    def __init__(self, use_cache: bool = False):
        # use_cache=True is for backfilling past days (data won't change).
        self.use_cache = use_cache

    def games_on(self, league: str, day: date) -> list[Game]:
        path = league_config(league)["espn_path"]
        data = get_json(
            f"{BASE}/{path}/scoreboard",
            params={"dates": day.strftime("%Y%m%d"), "limit": 200},
            cache=RAW_DIR / "espn" / league / "scoreboard" / f"{day.isoformat()}.json",
            use_cache=self.use_cache,
        )
        return [self._parse(league, ev) for ev in data.get("events", [])]

    def _parse(self, league: str, ev: dict) -> Game:
        comp = ev["competitions"][0]
        side = {c["homeAway"]: c for c in comp["competitors"]}
        status = comp.get("status") or ev.get("status") or {}
        done = bool(status.get("type", {}).get("completed"))
        season = ev.get("season", {})
        return Game(
            league=league,
            source_id=str(ev["id"]),
            start_time=to_iso(parse_utc(ev["date"])),
            season=int(season.get("year", 0)),
            season_type=SEASON_TYPES.get(season.get("type"), "other"),
            home_team=normalize(league, side["home"]["team"].get("abbreviation", "")),
            away_team=normalize(league, side["away"]["team"].get("abbreviation", "")),
            home_score=_score(side["home"], done),
            away_score=_score(side["away"], done),
            status=status.get("type", {}).get("name", "STATUS_UNKNOWN"),
        )


class EspnDetail(DetailSource):
    """Reads one game's summary page: odds from several books, the injury
    report, and (for finished games) who played how many minutes."""
    name = "espn"

    def __init__(self, use_cache: bool = False):
        self.use_cache = use_cache

    def detail(self, game: Game) -> GameDetail:
        if not game.source_id:
            raise ValueError("ESPN detail needs an ESPN event id")
        path = league_config(game.league)["espn_path"]
        data = get_json(
            f"{BASE}/{path}/summary",
            params={"event": game.source_id},
            cache=RAW_DIR / "espn" / game.league / "summary" / f"{game.source_id}.json",
            use_cache=self.use_cache and game.is_final,   # only finished games are safe to cache
        )
        return self.parse(game, data)

    def parse(self, game: Game, data: dict) -> GameDetail:
        return GameDetail(
            odds=self._odds(game, data),
            injuries=self._injuries(game, data),
            minutes=self._minutes(game, data) if game.is_final else [],
        )

    # ---- odds -------------------------------------------------------------
    def _odds(self, game: Game, data: dict) -> list[OddsSnapshot]:
        out = []
        current_kind = "close" if game.is_final else "live"
        for entry in data.get("pickcenter") or data.get("odds") or []:
            book = (entry.get("provider") or {}).get("name") or "unknown"
            home = parse_ml((entry.get("homeTeamOdds") or {}).get("moneyLine"))
            away = parse_ml((entry.get("awayTeamOdds") or {}).get("moneyLine"))
            if home and away:
                out.append(OddsSnapshot(game.game_key, book, home, away, current_kind, self.name))

            # Newer format (2024+): explicit open/close moneylines per side.
            ml = entry.get("moneyline") or {}
            for kind in ("open", "close"):
                h = parse_ml(((ml.get("home") or {}).get(kind) or {}).get("odds"))
                a = parse_ml(((ml.get("away") or {}).get(kind) or {}).get("odds"))
                if h and a and (kind == "open" or game.is_final):
                    out.append(OddsSnapshot(game.game_key, book, h, a, kind, self.name))
        return out

    # ---- injuries ---------------------------------------------------------
    def _injuries(self, game: Game, data: dict) -> list[InjuryReport]:
        out = []
        for team_block in data.get("injuries") or []:
            team = normalize(game.league, (team_block.get("team") or {}).get("abbreviation", ""))
            for inj in team_block.get("injuries") or []:
                athlete = inj.get("athlete") or {}
                status = inj.get("status") or (inj.get("type") or {}).get("description") or ""
                out.append(InjuryReport(
                    game_key=game.game_key,
                    team=team,
                    player=athlete.get("displayName", "unknown"),
                    status=str(status).title(),
                    position=((athlete.get("position") or {}).get("abbreviation") or ""),
                    source=self.name,
                ))
        return out

    # ---- box score minutes (NBA) -----------------------------------------
    def _minutes(self, game: Game, data: dict) -> list[PlayerMinutes]:
        out = []
        for team_block in (data.get("boxscore") or {}).get("players") or []:
            team = normalize(game.league, (team_block.get("team") or {}).get("abbreviation", ""))
            for stat_group in team_block.get("statistics") or []:
                labels = [str(x).upper() for x in stat_group.get("labels") or stat_group.get("names") or []]
                keys = [str(x).lower() for x in stat_group.get("keys") or []]
                if "MIN" in labels:
                    idx = labels.index("MIN")
                elif "minutes" in keys:
                    idx = keys.index("minutes")
                else:
                    continue
                for a in stat_group.get("athletes") or []:
                    stats = a.get("stats") or []
                    mins = 0.0 if a.get("didNotPlay") else (_minutes(stats[idx]) if idx < len(stats) else None)
                    if mins is None:
                        mins = 0.0
                    out.append(PlayerMinutes(game.game_key, team,
                                             (a.get("athlete") or {}).get("displayName", "unknown"), mins))
                break   # one stat group with minutes is enough
        return out
