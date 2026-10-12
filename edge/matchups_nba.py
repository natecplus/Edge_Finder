"""NBA player matchups: defense vs position (DvP).

For every defense we measure what it allows to guards, forwards and centers,
per 36 minutes, over its last 20 games, and compare that with the league
average for the same position. A player's projection is his own recent
per-minute production x his expected minutes x that defense's factor.

Small samples are shrunk toward average (a defense needs a few games of data
before it can look extreme), so early-season factors stay close to 1.0.
"""
import pandas as pd

from edge import db

STATS = ["pts", "reb", "ast", "fg3m"]
LABEL = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "3-pointers"}
GROUP_NAME = {"G": "guards", "F": "forwards", "C": "centers"}
WINDOW = 20          # defense games
PLAYER_GAMES = 15    # player games for per-minute rates
SHRINK_36 = 30       # pseudo "36-minute units" of league-average data added to each defense


def pos_group(pos: str) -> str:
    pos = (pos or "").upper()
    if "C" in pos:
        return "C"
    if pos.startswith("G") or pos in ("PG", "SG"):
        return "G"
    return "F"


def _with_opponent(stats: pd.DataFrame) -> pd.DataFrame:
    s = stats[stats.minutes > 0].copy()
    s["opponent"] = s.home_team.where(s.team != s.home_team, s.away_team)
    s["group"] = s.position.map(pos_group)
    return s


def defense_table(stats: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """One row per (defense, position group): factor per stat vs league average, and rank.
    factor 1.15 = allows 15% more than average; rank 1 = stingiest."""
    s = _with_opponent(stats[stats.start_time < as_of])
    if s.empty:
        return pd.DataFrame()
    # each defense's last WINDOW games
    recent = s[["opponent", "game_key", "start_time"]].drop_duplicates()
    recent = recent.sort_values("start_time").groupby("opponent").tail(WINDOW)
    s = s.merge(recent[["opponent", "game_key"]], on=["opponent", "game_key"])
    league = s.groupby("group")[STATS + ["minutes"]].sum()
    league_rate = league[STATS].div(league.minutes / 36, axis=0)       # per 36, by group
    agg = s.groupby(["opponent", "group"])[STATS + ["minutes"]].sum().reset_index()
    rows = []
    for r in agg.itertuples(index=False):
        units = r.minutes / 36
        row = {"defense": r.opponent, "group": r.group, "games": int(recent[recent.opponent == r.opponent].shape[0])}
        for st in STATS:
            lr = league_rate.loc[r.group, st]
            shrunk = (getattr(r, st) + SHRINK_36 * lr) / (units + SHRINK_36)
            row[f"{st}_factor"] = shrunk / lr if lr else 1.0
        rows.append(row)
    out = pd.DataFrame(rows)
    for st in STATS:
        out[f"{st}_rank"] = out.groupby("group")[f"{st}_factor"].rank(method="min").astype(int)
    return out


def player_table(stats: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """Recent per-minute rates, expected minutes and current team for every player."""
    s = stats[(stats.start_time < as_of) & (stats.minutes > 0)].sort_values("start_time")
    if s.empty:
        return pd.DataFrame()
    last = s.groupby("player").tail(PLAYER_GAMES)
    agg = last.groupby("player")[STATS + ["minutes"]].sum()
    rates = agg[STATS].div(agg.minutes, axis=0)
    info = s.groupby("player").agg(team=("team", "last"), position=("position", "last"),
                                   last_game=("start_time", "max"))
    mins = s.groupby("player").tail(5).groupby("player").minutes.mean().rename("proj_min")
    games = last.groupby("player").size().rename("games")
    out = info.join(rates.add_suffix("_rate")).join(mins).join(games).reset_index()
    out["group"] = out.position.map(pos_group)
    return out


class NbaContext:
    """Box scores loaded once; defense and player tables cached per cut-off time.
    Upcoming games all share the same cut-off (now), so a week of reports is fast."""

    def __init__(self, stats: pd.DataFrame | None = None):
        self.stats = db.player_stats_df("nba") if stats is None else stats
        self.games = db.games_df("nba").set_index("game_key")
        self._cache = {}

    def tables(self, as_of: str):
        if as_of not in self._cache:
            self._cache[as_of] = (defense_table(self.stats, as_of), player_table(self.stats, as_of))
        return self._cache[as_of]


def game_report(league: str, game_key: str, ctx: "NbaContext | pd.DataFrame | None" = None,
                min_minutes: float = 15) -> dict:
    if not isinstance(ctx, NbaContext):
        ctx = NbaContext(ctx)
    g = ctx.games.loc[game_key]
    # only games played before this one count; for upcoming games that's "everything so far"
    from edge.timeutil import utcnow, to_iso
    as_of = min(g.start_time, to_iso(utcnow()))
    dvp, players = ctx.tables(as_of)
    out = {"game_key": game_key, "sides": [], "data_note": ""}
    if dvp.empty or players.empty:
        out["data_note"] = "No box-score stats yet. Run: python -m scripts.load_matchup_data --league nba"
        return out
    injured = _out_players(game_key)
    for team, opp in ((g.away_team, g.home_team), (g.home_team, g.away_team)):
        d = dvp[dvp.defense == opp].set_index("group")
        roster = players[(players.team == team) & (players.proj_min >= min_minutes)
                         & ~players.player.isin(injured)]
        rows = []
        for p in roster.itertuples(index=False):
            if p.group not in d.index:
                continue
            f = d.loc[p.group]
            row = {"team": team, "opponent": opp, "player": p.player, "position": p.position,
                   "minutes": p.proj_min}
            for st in STATS:
                base = getattr(p, f"{st}_rate") * p.proj_min
                row[f"{st}_base"] = base
                row[f"{st}_proj"] = base * f[f"{st}_factor"]
                row[f"{st}_factor"] = f[f"{st}_factor"]
            row["advantage"] = f["pts_factor"] - 1
            row["metric"] = "Points"
            row["projection"] = row["pts_proj"]
            row["baseline"] = row["pts_base"]
            row["reasons"] = _reasons(opp, p.group, f, len(d))
            row["grade"] = _grade(row["advantage"])
            rows.append(row)
        table = pd.DataFrame(rows).sort_values("advantage", ascending=False) if rows else pd.DataFrame()
        profile = _defense_profile(opp, d)
        out["sides"].append({"offense": team, "defense": opp, "players": table, "defense_profile": profile})
    out["data_note"] = f"Defense factors from each team's last {WINDOW} games; player rates from last {PLAYER_GAMES}."
    return out


def _grade(adv: float) -> str:
    from edge.matchups_nfl import _grade as grade
    return grade(adv)


def _reasons(opp, group, f, n_groups) -> list[str]:
    out = []
    for st in ("pts", "reb", "ast", "fg3m"):
        fac = f[f"{st}_factor"]
        if abs(fac - 1) >= 0.05:
            rank = int(f[f"{st}_rank"])
            word = "more" if fac > 1 else "fewer"
            out.append(f"{opp} allows {abs(fac - 1):.0%} {word} {LABEL[st]} to {GROUP_NAME[group]} than average "
                       f"(rank {rank} of 30, 1 = toughest)")
    return out or [f"{opp} defends {GROUP_NAME[group]} about league average"]


def _defense_profile(opp: str, d: pd.DataFrame) -> list[str]:
    lines = []
    for grp in ("G", "F", "C"):
        if grp in d.index:
            fac = d.loc[grp, "pts_factor"]
            lines.append(f"vs {GROUP_NAME[grp]}: {fac - 1:+.0%} points allowed vs average "
                         f"(rank {int(d.loc[grp, 'pts_rank'])})")
    return lines


def _out_players(game_key: str) -> set:
    inj = db.read_sql("SELECT player, status, captured_at FROM injuries WHERE game_key = :k", k=game_key)
    if inj.empty:
        return set()
    latest = inj[inj.captured_at == inj.captured_at.max()]
    return set(latest[latest.status.isin(["Out", "Doubtful", "Suspension"])].player)
