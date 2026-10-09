"""Tennis features: one row per match, everything known before the first serve.

  elo_diff          overall Elo (player A minus player B)
  surface_elo_diff  Elo on this surface only (hard / clay / grass behave differently)
  rank_diff         log2(B's ranking / A's ranking): +1 means A is ranked twice as high
  form_diff         win rate over each player's last 10 matches
  fatigue_diff      matches played in the previous 7 days
  rest_diff         days since each player's last match (capped at 14)

"A" is the "home" column, i.e. the player first in alphabetical order, so the
sign of a feature never hints at the winner. Players are identified by
players.player_key, so "Zhang Zh." and "Zhang Z." are the same person.
"""
import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from edge.players import player_key

TENNIS_FEATURES = ["elo_diff", "surface_elo_diff", "rank_diff", "form_diff", "fatigue_diff", "rest_diff"]
UNRANKED = 1500


def _k(n_matches: int) -> float:
    # big moves for new players, smaller for established ones (538's tennis Elo)
    return 250 / (n_matches + 5) ** 0.4


def _expected(a: float, b: float) -> float:
    return 1 / (1 + 10 ** ((b - a) / 400))


def build_tennis_features(games: pd.DataFrame, cfg: dict, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    df = games.copy()
    df["home_score"] = pd.to_numeric(df.home_score, errors="coerce")
    df["away_score"] = pd.to_numeric(df.away_score, errors="coerce")
    if meta is None:
        from edge import db
        meta = db.tennis_meta_df(cfg["league"])
    keep = ["game_key", "tournament", "surface", "round", "series", "home_rank", "away_rank", "comment"]
    meta = meta.reindex(columns=keep).drop_duplicates("game_key")
    df = df.merge(meta, on="game_key", how="left")
    df["surface"] = df.surface.fillna("Hard").replace({"": "Hard", "nan": "Hard"})
    df["t_start"] = pd.to_datetime(df.start_time, utc=True)
    df = df.sort_values(["t_start", "game_key"]).reset_index(drop=True)

    elo = defaultdict(lambda: 1500.0)
    surf = defaultdict(lambda: 1500.0)                  # key: (player, surface)
    n_played = defaultdict(int)
    n_surf = defaultdict(int)
    recent = defaultdict(lambda: deque(maxlen=cfg.get("form_window", 10)))
    history = defaultdict(list)                         # match times
    last_rank = {}
    season_count = defaultdict(int)                     # (player, season)

    cols = defaultdict(list)
    for r in df.itertuples(index=False):
        a, b = player_key(r.home_team), player_key(r.away_team)
        s = r.surface
        # ranking going into the match: this row's if known, else the latest seen before
        ra = r.home_rank if pd.notna(r.home_rank) else last_rank.get(a)
        rb = r.away_rank if pd.notna(r.away_rank) else last_rank.get(b)
        week_ago = r.t_start - pd.Timedelta(days=7)
        for side, p in (("home", a), ("away", b)):
            h = history[p]
            cols[f"{side}_elo"].append(elo[p])
            cols[f"{side}_surf_elo"].append(surf[(p, s)])
            cols[f"form_{side}"].append(np.mean(recent[p]) if len(recent[p]) >= 3 else np.nan)
            cols[f"fat_{side}"].append(sum(1 for t in h[-10:] if week_ago <= t < r.t_start))
            cols[f"rest_{side}"].append(min((r.t_start - h[-1]).total_seconds() / 86400, 14) if h else 14.0)
            cols[f"gp_{side}"].append(season_count[(p, r.season)])
        cols["rank_home"].append(ra)
        cols["rank_away"].append(rb)

        played = pd.notna(r.home_score) and pd.notna(r.away_score) and r.home_score != r.away_score \
            and str(r.comment) != "Walkover"
        if not played:
            continue
        res = 1.0 if r.home_score > r.away_score else 0.0
        e = _expected(elo[a], elo[b])
        ka, kb = _k(n_played[a]), _k(n_played[b])
        elo[a] += ka * (res - e)
        elo[b] -= kb * (res - e)
        es = _expected(surf[(a, s)], surf[(b, s)])
        sa, sb = _k(n_surf[(a, s)]), _k(n_surf[(b, s)])
        surf[(a, s)] += sa * (res - es)
        surf[(b, s)] -= sb * (res - es)
        for p, won in ((a, res), (b, 1 - res)):
            n_played[p] += 1
            n_surf[(p, s)] += 1
            recent[p].append(won)
            history[p].append(r.t_start)
            season_count[(p, r.season)] += 1
        if pd.notna(r.home_rank):
            last_rank[a] = int(r.home_rank)
        if pd.notna(r.away_rank):
            last_rank[b] = int(r.away_rank)

    for c, v in cols.items():
        df[c] = v
    df["elo_diff"] = df.home_elo - df.away_elo
    df["surface_elo_diff"] = df.home_surf_elo - df.away_surf_elo
    rh = pd.to_numeric(df.rank_home, errors="coerce").fillna(UNRANKED).clip(lower=1)
    ra = pd.to_numeric(df.rank_away, errors="coerce").fillna(UNRANKED).clip(lower=1)
    df["rank_diff"] = np.log2(ra / rh)
    df["form_diff"] = df.form_home.fillna(0.5) - df.form_away.fillna(0.5)
    df["fatigue_diff"] = df.fat_home - df.fat_away
    df["rest_diff"] = df.rest_home - df.rest_away
    # columns the shared code (rules, app) expects
    df["b2b_home"] = 0
    df["b2b_away"] = 0
    df["miss_home"] = 0.0
    df["miss_away"] = 0.0
    df["season_margin_diff"] = 0.0

    finished = df.home_score.notna() & df.away_score.notna() & (df.comment.astype(str) != "Walkover")
    df["home_win"] = np.where(finished, (df.home_score > df.away_score).astype(float), np.nan)
    df.loc[finished & (df.home_score == df.away_score), "home_win"] = np.nan
    return df.drop(columns=["t_start"])


def rank_text(v) -> str:
    return "unranked" if v is None or (isinstance(v, float) and math.isnan(v)) else f"#{int(v)}"
