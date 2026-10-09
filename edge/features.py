"""Turn raw games into one row of model inputs per game.

The golden rule: every feature for a game is computed only from things known
BEFORE that game started. tests/test_features.py checks this automatically.

The one deliberate exception is documented below (availability): for past
games we use who actually played as a stand-in for the pregame injury report.
"""
import numpy as np
import pandas as pd

from edge.elo import run_elo

OUT_STATUSES = {"Out", "Doubtful", "Suspension", "Suspended", "Injured Reserve", "Ir"}
REGULAR_MINUTES = 10.0     # players averaging fewer minutes don't count as "rotation"


def feature_list(cfg: dict, kind: str) -> list[str]:
    if kind == "elo_only":
        return ["elo_diff"]
    if cfg.get("sport") == "tennis":
        from edge.tennis_features import TENNIS_FEATURES
        return list(TENNIS_FEATURES)
    cols = ["elo_diff", "form_diff", "season_margin_diff", "rest_diff", "b2b_home", "b2b_away"]
    if cfg.get("use_minutes"):
        cols += ["miss_home", "miss_away"]
    return cols


# ------------------------------------------------------------------ team log
def team_log(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game (each game appears twice: home and away)."""
    base = ["game_key", "season", "start_time"]
    home = games[base].assign(team=games.home_team, pts=games.home_score,
                              opp_pts=games.away_score, is_home=1)
    away = games[base].assign(team=games.away_team, pts=games.away_score,
                              opp_pts=games.home_score, is_home=0)
    log = pd.concat([home, away], ignore_index=True)
    log["start_time"] = pd.to_datetime(log["start_time"], utc=True)
    return log.sort_values(["team", "start_time"]).reset_index(drop=True)


def add_form(log: pd.DataFrame, window: int) -> pd.DataFrame:
    log = log.copy()
    log["margin"] = log.pts - log.opp_pts            # NaN for unplayed games
    by_team = log.groupby("team", group_keys=False)
    # shift(1): a game only ever sees the games BEFORE it
    log["form"] = by_team["margin"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=2).mean())
    log["season_margin"] = log.groupby(["team", "season"], group_keys=False)["margin"].transform(
        lambda s: s.shift(1).expanding().mean())
    rest = by_team["start_time"].diff().dt.total_seconds() / 86400
    log["rest"] = rest.clip(upper=7).fillna(7)
    log["games_played"] = log.groupby(["team", "season"])["margin"].transform(
        lambda s: s.shift(1).notna().cumsum()).fillna(0).astype(int)
    return log


# --------------------------------------------------------------- availability
def usual_minutes(minutes: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    """For every (team, game), each player's average minutes over the team's
    previous 10 played games (0 when he didn't play). Returns long format:
    game_key, team, player, usual, played_min."""
    if minutes.empty:
        return pd.DataFrame(columns=["game_key", "team", "player", "usual", "played_min"])
    played = log[log.margin.notna()][["team", "game_key", "start_time"]]
    out = []
    for team, tgames in played.groupby("team"):
        m = minutes[minutes.team == team]
        if m.empty:
            continue
        wide = (m.pivot_table(index="game_key", columns="player", values="minutes", aggfunc="sum")
                .reindex(tgames.sort_values("start_time").game_key).fillna(0.0))
        usual = wide.shift(1).rolling(10, min_periods=3).mean()
        stacked = usual.stack().rename("usual").reset_index()
        stacked.columns = ["game_key", "player", "usual"]
        played_min = wide.stack().rename("played_min").reset_index()
        played_min.columns = ["game_key", "player", "played_min"]
        merged = stacked.merge(played_min, on=["game_key", "player"], how="left")
        merged["team"] = team
        out.append(merged)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["game_key", "team", "player", "usual", "played_min"])


def missing_share_history(um: pd.DataFrame) -> pd.DataFrame:
    """Share of a team's usual rotation minutes that did NOT play in a past game.

    Train/serve note: for past games we use actual absences (box score), while
    for upcoming games we use the injury report. They agree except for late
    scratches; that small mismatch is accepted and documented in the README."""
    if um.empty:
        return pd.DataFrame(columns=["game_key", "team", "miss"])
    um = um[um.usual >= REGULAR_MINUTES].copy()
    um["missing"] = np.where(um.played_min <= 0, um.usual, 0.0)
    agg = um.groupby(["game_key", "team"]).agg(missing=("missing", "sum"), total=("usual", "sum"))
    agg["miss"] = (agg.missing / agg.total).fillna(0.0)
    return agg[["miss"]].reset_index()


def latest_usual(minutes: pd.DataFrame, log: pd.DataFrame, team: str, before: str) -> pd.Series:
    """Each player's average minutes over the team's last 10 games played BEFORE `before`."""
    cutoff = pd.Timestamp(before)
    played = log[(log.team == team) & log.margin.notna() & (log.start_time < cutoff)] \
        .sort_values("start_time").tail(10)
    m = minutes[(minutes.team == team) & minutes.game_key.isin(played.game_key)]
    if m.empty or len(played) < 3:
        return pd.Series(dtype=float)
    wide = m.pivot_table(index="game_key", columns="player", values="minutes", aggfunc="sum") \
            .reindex(played.game_key).fillna(0.0)
    return wide.mean().sort_values(ascending=False)


def upcoming_availability(game_key: str, team: str, start_time: str, minutes: pd.DataFrame,
                          log: pd.DataFrame, injuries: pd.DataFrame) -> dict:
    """Missing-rotation share for an upcoming game, from the latest injury report
    captured before tip-off."""
    result = {"miss": 0.0, "out_players": [], "top_player_out": None, "qb_out": []}
    inj = injuries[(injuries.game_key == game_key) & (injuries.team == team)]
    if not inj.empty:
        inj = inj[inj.captured_at <= start_time]
        if not inj.empty:
            inj = inj[inj.captured_at == inj.captured_at.max()]
    out = inj[inj.status.isin(OUT_STATUSES)] if not inj.empty else inj
    result["out_players"] = list(out.player) if not out.empty else []
    if not out.empty and "position" in out:
        result["qb_out"] = list(out[out.position.str.upper() == "QB"].player)
    usual = latest_usual(minutes, log, team, start_time) if not minutes.empty else pd.Series(dtype=float)
    usual = usual[usual >= REGULAR_MINUTES]
    if usual.empty:
        return result
    missing = usual[usual.index.isin(result["out_players"])]
    result["miss"] = float(missing.sum() / usual.sum())
    if usual.index[0] in result["out_players"]:
        result["top_player_out"] = usual.index[0]
    return result


# ------------------------------------------------------------------ main entry
def build_features(games: pd.DataFrame, cfg: dict, minutes: pd.DataFrame | None = None,
                   injuries: pd.DataFrame | None = None, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per game with features and (for finished games) the label home_win."""
    if cfg.get("sport") == "tennis":
        from edge.tennis_features import build_tennis_features
        return build_tennis_features(games, cfg, meta)
    games = games.copy()
    games["home_score"] = pd.to_numeric(games.home_score, errors="coerce")
    games["away_score"] = pd.to_numeric(games.away_score, errors="coerce")
    minutes = minutes if minutes is not None else pd.DataFrame(columns=["game_key", "team", "player", "minutes"])
    injuries = injuries if injuries is not None else pd.DataFrame(
        columns=["game_key", "team", "player", "status", "position", "captured_at"])

    pregame, current = run_elo(games, k=cfg.get("elo_k", 20), home_adv=cfg.get("home_advantage_elo", 70))
    df = games.join(pregame, on="game_key")
    # Preseason games aren't in the Elo replay; give them current ratings (display only).
    df["home_elo"] = df.home_elo.fillna(df.home_team.map(current)).fillna(1500.0)
    df["away_elo"] = df.away_elo.fillna(df.away_team.map(current)).fillna(1500.0)
    df["elo_diff"] = df.home_elo - df.away_elo

    log = add_form(team_log(games[games.season_type != "preseason"]), cfg.get("form_window", 10))
    cols = ["game_key", "team", "form", "season_margin", "rest", "games_played"]
    home = log[log.is_home == 1][cols].rename(columns=lambda c: c if c == "game_key" else f"{c}_home")
    away = log[log.is_home == 0][cols].rename(columns=lambda c: c if c == "game_key" else f"{c}_away")
    df = df.merge(home.drop(columns="team_home"), on="game_key", how="left") \
           .merge(away.drop(columns="team_away"), on="game_key", how="left")

    df["form_diff"] = (df.form_home.fillna(0) - df.form_away.fillna(0))
    df["season_margin_diff"] = df.season_margin_home.fillna(0) - df.season_margin_away.fillna(0)
    df["rest_home"] = df.rest_home.fillna(7)
    df["rest_away"] = df.rest_away.fillna(7)
    df["rest_diff"] = df.rest_home - df.rest_away
    short = 1.5 if cfg["league"] == "nba" else 5.5        # back-to-back / short week
    df["b2b_home"] = (df.rest_home <= short).astype(int)
    df["b2b_away"] = (df.rest_away <= short).astype(int)
    df["gp_home"] = df.games_played_home.fillna(0).astype(int)
    df["gp_away"] = df.games_played_away.fillna(0).astype(int)

    # Availability: history from box scores, upcoming games from the injury report.
    df["miss_home"] = 0.0
    df["miss_away"] = 0.0
    if cfg.get("use_minutes") and not minutes.empty:
        hist = missing_share_history(usual_minutes(minutes, log))
        miss = hist.set_index(["game_key", "team"])["miss"]
        df["miss_home"] = [miss.get((k, t), 0.0) for k, t in zip(df.game_key, df.home_team)]
        df["miss_away"] = [miss.get((k, t), 0.0) for k, t in zip(df.game_key, df.away_team)]

    finished = df.home_score.notna() & df.away_score.notna()
    df["home_win"] = np.where(finished, (df.home_score > df.away_score).astype(float), np.nan)
    df.loc[finished & (df.home_score == df.away_score), "home_win"] = np.nan   # rare NFL ties: drop
    df.attrs["team_log"] = log
    return df


def fill_upcoming_availability(df: pd.DataFrame, cfg: dict, minutes: pd.DataFrame,
                               injuries: pd.DataFrame, keys: list[str] | None = None) -> dict:
    """Overwrite miss_home/miss_away using the pregame injury report, for unplayed
    games (default) or for the given game keys (used when replaying past days,
    so a replayed pick only knows what the injury report said at the time).
    Returns per-game details for the rules engine and the game-detail page."""
    if cfg.get("sport") == "tennis":
        return {}                     # no injury reports in tennis: withdrawals remove the match
    log = df.attrs.get("team_log")
    details = {}
    if keys is not None:
        upcoming = df[df.game_key.isin(keys)]
    else:
        upcoming = df[df.home_win.isna() & df.home_score.isna()]
    for idx, g in upcoming.iterrows():
        info = {}
        for side in ("home", "away"):
            team = g[f"{side}_team"]
            a = upcoming_availability(g.game_key, team, g.start_time, minutes, log, injuries)
            info[side] = a
            if cfg.get("use_minutes"):
                df.at[idx, f"miss_{side}"] = a["miss"]
        details[g.game_key] = info
    return details
