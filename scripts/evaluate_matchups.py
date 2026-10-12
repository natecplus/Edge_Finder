"""Does the matchup score actually predict anything? Walk-forward check.

    python -m scripts.evaluate_matchups --league nfl --weeks 6     # last 6 weeks of data
    python -m scripts.evaluate_matchups --league nba --days 30

For each past week (NFL) or day (NBA), the report is rebuilt using ONLY data
from before it, then compared with what actually happened:
  corr  = correlation between the advantage % and the player's actual result
          vs his baseline (above 0 = the score points the right way)
  MAE   = average miss of the matchup projection vs the plain baseline
          (projection lower = the matchup adjustment helped)
"""
import argparse

import numpy as np
import pandas as pd


def _summary(rows: list[tuple]) -> None:
    df = pd.DataFrame(rows, columns=["metric", "adv", "actual", "proj", "base"])
    if df.empty:
        print("Not enough data to evaluate.")
        return
    df["rel"] = np.where(df.base > 0, df.actual / df.base - 1, np.nan)
    print(f"{'metric':<16}{'n':>6}{'corr':>8}{'MAE proj':>10}{'MAE base':>10}")
    for m, g in df.groupby("metric"):
        corr = g[["adv", "rel"]].corr().iloc[0, 1]
        print(f"{m:<16}{len(g):>6}{corr:>8.3f}{(g.proj - g.actual).abs().mean():>10.1f}"
              f"{(g.base - g.actual).abs().mean():>10.1f}")


def nfl(weeks: int) -> None:
    from edge.matchups_nfl import NflContext, default_seasons
    from edge.sources import nflverse
    from edge.teams import normalize
    data = nflverse.load(default_seasons())
    pbp = data["pbp"]
    if pbp.empty:
        print("No play-by-play. Run: python -m scripts.load_matchup_data --league nfl")
        return
    season = int(pbp.season.max())
    last_week = int(pbp[pbp.season == season].week.max())
    rows = []
    for wk in range(max(2, last_week - weeks + 1), last_week + 1):
        before = {**data, "pbp": pbp[(pbp.season < season) | (pbp.week < wk)]}
        ctx = NflContext(data=before)
        test = pbp[(pbp.season == season) & (pbp.week == wk)]
        for off, dfn in test[["posteam", "defteam"]].drop_duplicates().dropna().itertuples(index=False):
            t = ctx.side(normalize("nfl", off), normalize("nfl", dfn))["players"]
            g = test[test.posteam == off]
            for r in t.itertuples() if not t.empty else []:
                if r.metric == "Receiving yards":
                    actual = g[g.receiver_player_name == r.player].yards_gained.sum()
                elif r.metric == "Rush yards":
                    actual = g[g.rusher_player_name == r.player].yards_gained.sum()
                else:
                    actual = g[(g["pass"] == 1) & (g.passer_player_name == r.player)].yards_gained.sum()
                rows.append((r.metric, r.advantage, actual, r.projection, r.baseline))
        print(f"  week {wk} done")
    _summary(rows)


def nba(days: int) -> None:
    from edge.matchups_nba import NbaContext, game_report
    ctx = NbaContext()
    if ctx.stats.empty:
        print("No box scores. Run: python -m scripts.load_matchup_data --league nba")
        return
    games = ctx.games[ctx.games.home_score.notna()].sort_values("start_time")
    cutoff = pd.Timestamp(games.start_time.max()) - pd.Timedelta(days=days)
    recent = games[pd.to_datetime(games.start_time) >= cutoff]
    actual = ctx.stats.set_index(["game_key", "player"]).pts
    rows = []
    for key in recent.index:
        for side in game_report("nba", key, ctx)["sides"]:
            for r in side["players"].itertuples() if not side["players"].empty else []:
                if (key, r.player) in actual.index:
                    rows.append(("Points", r.advantage, float(actual.loc[(key, r.player)]), r.projection, r.baseline))
    _summary(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True, choices=["nfl", "nba"])
    ap.add_argument("--weeks", type=int, default=6)
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()
    nfl(args.weeks) if args.league == "nfl" else nba(args.days)


if __name__ == "__main__":
    main()
