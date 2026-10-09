"""ESPN adapter for ATP singles (today's matches + live results).

ESPN's tennis scoreboard is shaped differently from team sports: each event is
a TOURNAMENT, holding "groupings" (Men's Singles, Men's Doubles, ...), each
holding "competitions" (the matches). Parsed defensively: if ESPN moves a
field, the fixture test in tests/test_tennis.py shows which one.
"""
from dataclasses import dataclass, field
from datetime import date

from edge.config import RAW_DIR, league_config
from edge.http import get_json
from edge.players import display_name, ordered
from edge.timeutil import parse_utc, to_iso
from .base import Game, OddsSnapshot
from .espn import BASE, parse_ml


GRAND_SLAMS = ("Australian Open", "Roland Garros", "French Open", "Wimbledon", "US Open")


@dataclass
class TennisMatch:
    game: Game
    meta: dict = field(default_factory=dict)       # tournament, surface, round, best_of
    odds: list[OddsSnapshot] = field(default_factory=list)


def _sets_won(me: dict, opp: dict) -> int:
    won = 0
    for a, b in zip(me.get("linescores") or [], opp.get("linescores") or []):
        try:
            if float(a.get("value", 0)) > float(b.get("value", 0)):
                won += 1
        except (TypeError, ValueError):
            pass
    return won


def _name(c: dict) -> str:
    a = c.get("athlete") or {}
    return a.get("displayName") or a.get("fullName") or (c.get("team") or {}).get("displayName") or ""


def _is_mens_singles(grouping: dict, comp: dict) -> bool:
    g = grouping.get("grouping") or {}
    label = " ".join(str(x) for x in (g.get("slug"), g.get("displayName"),
                                      (comp.get("type") or {}).get("text"),
                                      (comp.get("type") or {}).get("slug")) if x).lower()
    if label:
        return "single" in label and "women" not in label and "double" not in label
    # no label: singles matches have one athlete per side, doubles a roster
    return all("athlete" in c for c in comp.get("competitors") or [])


def parse_scoreboard(data: dict, league: str = "atp") -> list[TennisMatch]:
    out = []
    for ev in data.get("events") or []:
        tournament = ev.get("name") or ev.get("shortName") or ""
        season = int((ev.get("season") or {}).get("year") or str(ev.get("date", "0000"))[:4])
        groupings = ev.get("groupings") or [{"competitions": ev.get("competitions") or []}]
        for grp in groupings:
            for comp in grp.get("competitions") or []:
                if not _is_mens_singles(grp, comp):
                    continue
                cs = comp.get("competitors") or []
                if len(cs) != 2 or not all(_name(c) for c in cs):
                    continue                       # TBD vs TBD, byes
                status = (comp.get("status") or {}).get("type") or {}
                done = bool(status.get("completed"))
                a, b = cs
                na, nb = display_name(_name(a)), display_name(_name(b))
                sa, sb = _sets_won(a, b), _sets_won(b, a)
                if done and sa == sb:              # retirement mid-set: credit the winner
                    sa, sb = (sa + 1, sb) if a.get("winner") else (sa, sb + 1)
                score = {na: sa, nb: sb}
                home, away = ordered(na, nb)
                start = comp.get("date") or comp.get("startDate") or ev.get("date")
                game = Game(
                    league=league, source_id=str(comp.get("id") or ""),
                    start_time=to_iso(parse_utc(start)), season=season, season_type="regular",
                    home_team=home, away_team=away,
                    home_score=score[home] if done else None,
                    away_score=score[away] if done else None,
                    status=status.get("name") or ("STATUS_FINAL" if done else "STATUS_SCHEDULED"),
                )
                comment = "Completed"
                detail = str(status.get("detail") or status.get("description") or "").lower()
                if "walkover" in detail or "w/o" in detail:
                    comment = "Walkover"
                elif "retire" in detail:
                    comment = "Retired"
                meta = {
                    "game_key": game.game_key, "tournament": tournament,
                    "round": (comp.get("round") or {}).get("displayName"),
                    "best_of": 5 if any(s in tournament for s in GRAND_SLAMS) else 3,
                    "comment": comment,
                }
                odds = []
                for o in comp.get("odds") or []:
                    # ESPN tags one competitor "home"; fall back to listing order
                    first = next((c for c in cs if c.get("homeAway") == "home"), a)
                    n_first = display_name(_name(first))
                    n_second = nb if n_first == na else na
                    side = {n_first: parse_ml((o.get("homeTeamOdds") or {}).get("moneyLine")),
                            n_second: parse_ml((o.get("awayTeamOdds") or {}).get("moneyLine"))}
                    if side.get(home) and side.get(away):
                        book = (o.get("provider") or {}).get("name") or "ESPN"
                        odds.append(OddsSnapshot(game.game_key, book, side[home], side[away], "live", "espn"))
                out.append(TennisMatch(game, meta, odds))
    return out


class EspnTennis:
    name = "espn"

    def __init__(self, use_cache: bool = False):
        self.use_cache = use_cache

    def matches_on(self, league: str, day: date) -> list[TennisMatch]:
        path = league_config(league)["espn_path"]
        data = get_json(
            f"{BASE}/{path}/scoreboard", params={"dates": day.strftime("%Y%m%d")},
            cache=RAW_DIR / "espn" / league / "scoreboard" / f"{day.isoformat()}.json",
            use_cache=self.use_cache)
        return parse_scoreboard(data, league)
