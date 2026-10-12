"""NFL player matchups: how each offensive player fits the defense he faces.

Built from nflverse play-by-play + FTN coverage charting (free files):

  Defense profile   man vs zone rate, most-used coverages (Cover 1/2/3/4/6...),
                    pressure rate, yards/target allowed to WRs, TEs and RBs,
                    yards/carry allowed, yards allowed by route
  Player profile    yards/target overall and vs man / vs zone, top routes,
                    target share, carries, QB yards/attempt clean vs pressured

  Receiver advantage = scheme fit (his man/zone splits weighted by how often
  this defense plays each) x defense weakness vs his position.
  RB advantage = run defense weakness. QB advantage = pass defense x scheme
  fit x pressure.

Every split is shrunk toward a sensible average (a WR with 9 targets vs man
doesn't get credit for a fluke), and last season counts at 35% weight so the
report works in September. It's a descriptive matchup tool, not a validated
projection model: treat it as a well-organized scouting report.
"""
import numpy as np
import pandas as pd

from edge import db
from edge.sources import nflverse
from edge.teams import normalize

PREV_SEASON_WEIGHT = 0.35
K_POS_TARGETS = 40       # defense vs position: pseudo-targets of league average
K_PLAYER_TARGETS = 30    # player yards/target toward his position's average
K_SPLIT = 25             # player man/zone split toward his own overall
K_CARRIES = 60
K_ATTEMPTS = 150
K_ROUTE = 15
RECEIVER_POS = ("WR", "TE", "RB")


def _grade(adv: float) -> str:
    if adv >= 0.12:
        return "Strong"
    if adv >= 0.05:
        return "Good"
    if adv <= -0.12:
        return "Tough"
    if adv <= -0.05:
        return "Below avg"
    return "Neutral"


def _shrink(total, n, prior, k):
    """(sum + k*prior) / (n + k): an average pulled toward `prior` when n is small."""
    return (total + k * prior) / (n + k)


class NflContext:
    """Everything computed once from the play-by-play, reused for every game."""

    def __init__(self, data: dict | None = None, seasons: list[int] | None = None):
        if data is None:
            data = nflverse.load(seasons or [])
        pbp = data["pbp"].copy()
        self.empty = pbp.empty
        if self.empty:
            return
        self.season = int(pbp.season.max())
        self.charted_seasons = []
        pbp = pbp[(pbp.get("two_point_attempt", 0).fillna(0) == 0) &
                  ((pbp["pass"].fillna(0) == 1) | (pbp["rush"].fillna(0) == 1))].copy()
        pbp["posteam"] = pbp.posteam.map(lambda t: normalize("nfl", str(t)))
        pbp["defteam"] = pbp.defteam.map(lambda t: normalize("nfl", str(t)))
        pbp["w"] = np.where(pbp.season == self.season, 1.0, PREV_SEASON_WEIGHT)
        pbp["yards_gained"] = pbp.yards_gained.fillna(0)

        part = data["participation"]
        if not part.empty:
            part = part.rename(columns={"nflverse_game_id": "game_id"})
            pbp = pbp.merge(part, on=["game_id", "play_id"], how="left")
            mz = pbp.defense_man_zone_type.astype(str).str.upper()
            pbp["coverage"] = np.where(mz.str.contains("MAN"), "man",
                                       np.where(mz.str.contains("ZONE"), "zone", None))
            self.charted_seasons = sorted(int(s) for s in pbp[pbp.coverage.notna()].season.unique())
        else:
            for c in ("coverage", "defense_coverage_type", "route", "was_pressure"):
                pbp[c] = None

        roster = data["roster"]
        pos = (roster.sort_values("season").drop_duplicates("gsis_id", keep="last")
               .set_index("gsis_id").position) if not roster.empty else pd.Series(dtype=str)
        pbp["target_pos"] = pbp.receiver_player_id.map(pos)
        self.pbp = pbp
        self._league()
        self.defense = self._defense()
        self.players = self._players()

    # ------------------------------------------------------------ league
    def _league(self):
        p = self.pbp
        tgt = p[p.receiver_player_id.notna()]
        self.lg_ypt = {pos: np.average(g.yards_gained, weights=g.w)
                       for pos, g in tgt.groupby("target_pos") if pos in RECEIVER_POS}
        self.lg_ypt_all = np.average(tgt.yards_gained, weights=tgt.w) if len(tgt) else 7.0
        runs = p[p.rush == 1]
        self.lg_ypc = np.average(runs.yards_gained, weights=runs.w) if len(runs) else 4.3
        att = p[(p["pass"] == 1) & (p.sack.fillna(0) == 0)]
        self.lg_ypa = np.average(att.yards_gained, weights=att.w) if len(att) else 6.5
        ch = p[(p.qb_dropback == 1) & p.coverage.notna()]
        self.lg_man = np.average(ch.coverage == "man", weights=ch.w) if len(ch) else 0.35
        pr = p[(p.qb_dropback == 1) & p.was_pressure.notna()]
        self.lg_pressure = np.average(pr.was_pressure.astype(float), weights=pr.w) if len(pr) else 0.3
        rt = tgt[tgt.route.notna()]
        self.lg_route = rt.groupby("route").apply(lambda g: np.average(g.yards_gained, weights=g.w),
                                                  include_groups=False).to_dict() if len(rt) else {}

    # ------------------------------------------------------------ defenses
    def _defense(self) -> pd.DataFrame:
        rows = []
        for team, d in self.pbp.groupby("defteam"):
            row = {"defense": team}
            drop = d[d.qb_dropback == 1]
            ch = drop[drop.coverage.notna()]
            row["charted"] = len(ch)
            row["man_rate"] = _shrink((ch.coverage == "man").mul(ch.w).sum(), ch.w.sum(), self.lg_man, 40)
            covs = ch.groupby("defense_coverage_type").w.sum()
            covs = covs[~covs.index.isin(["BLOWN", "None", "nan"])]
            row["top_coverages"] = (covs / covs.sum()).sort_values(ascending=False).head(3).to_dict() if len(covs) else {}
            pr = drop[drop.was_pressure.notna()]
            row["pressure_rate"] = _shrink(pr.was_pressure.astype(float).mul(pr.w).sum(), pr.w.sum(), self.lg_pressure, 60)
            tgt = d[d.receiver_player_id.notna()]
            for pos in RECEIVER_POS:
                t = tgt[tgt.target_pos == pos]
                lg = self.lg_ypt.get(pos, self.lg_ypt_all)
                row[f"ypt_{pos}"] = _shrink(t.yards_gained.mul(t.w).sum(), t.w.sum(), lg, K_POS_TARGETS)
            runs = d[d.rush == 1]
            row["ypc"] = _shrink(runs.yards_gained.mul(runs.w).sum(), runs.w.sum(), self.lg_ypc, K_CARRIES)
            att = d[(d["pass"] == 1) & (d.sack.fillna(0) == 0)]
            row["ypa"] = _shrink(att.yards_gained.mul(att.w).sum(), att.w.sum(), self.lg_ypa, K_ATTEMPTS)
            rt = tgt[tgt.route.notna()]
            row["route_ypt"] = {r: _shrink(g.yards_gained.mul(g.w).sum(), g.w.sum(), self.lg_route.get(r, self.lg_ypt_all), K_ROUTE)
                                for r, g in rt.groupby("route")}
            rows.append(row)
        df = pd.DataFrame(rows).set_index("defense")
        n = len(df)
        for col in [f"ypt_{p}" for p in RECEIVER_POS] + ["ypc", "ypa"]:
            df[f"{col}_rank"] = df[col].rank(method="min").astype(int)      # 1 = allows least
        df.attrs["n"] = n
        return df

    # ------------------------------------------------------------ players
    def _players(self) -> pd.DataFrame:
        p = self.pbp
        cur = p[p.season == self.season]
        team_games = cur.groupby("posteam").game_id.nunique()
        team_att = cur[cur["pass"] == 1].groupby("posteam").size() / team_games
        rows = []

        # receivers
        tgt = p[p.receiver_player_id.notna()]
        for pid, g in tgt.groupby("receiver_player_id"):
            last = g.sort_values(["season", "week"]).iloc[-1]
            if last.season != self.season:
                continue                               # not active this season
            pos = last.target_pos if isinstance(last.target_pos, str) else "WR"
            if pos not in RECEIVER_POS:
                continue
            team = last.posteam
            lg = self.lg_ypt.get(pos, self.lg_ypt_all)
            y = _shrink(g.yards_gained.mul(g.w).sum(), g.w.sum(), lg, K_PLAYER_TARGETS)
            splits = {}
            for cov in ("man", "zone"):
                s = g[g.coverage == cov]
                splits[cov] = _shrink(s.yards_gained.mul(s.w).sum(), s.w.sum(), y, K_SPLIT)
                splits[f"n_{cov}"] = len(s)
            gc = g[g.season == self.season]
            share = len(gc) / max(cur[(cur.posteam == team) & (cur["pass"] == 1)].shape[0], 1)
            routes = g[g.route.notna()].groupby("route").agg(n=("w", "size"), y=("yards_gained", "mean"))
            routes = routes[routes.n >= 5].sort_values("n", ascending=False).head(2)
            rows.append({"player_id": pid, "player": last.receiver_player_name, "team": team, "position": pos,
                         "kind": "receiver", "ypt": y, "ypt_man": splits["man"], "ypt_zone": splits["zone"],
                         "n_man": splits["n_man"], "n_zone": splits["n_zone"], "targets": len(gc),
                         "proj_volume": share * team_att.get(team, 33), "routes": routes.to_dict("index")})

        # rushers
        runs = p[(p.rush == 1) & p.rusher_player_id.notna()]
        for pid, g in runs.groupby("rusher_player_id"):
            last = g.sort_values(["season", "week"]).iloc[-1]
            if last.season != self.season:
                continue
            gc = g[g.season == self.season]
            team = last.posteam
            per_game = len(gc) / max(team_games.get(team, 1), 1)
            if per_game < 5:
                continue                               # skip QB scrambles / gadget carries
            ypc = _shrink(g.yards_gained.mul(g.w).sum(), g.w.sum(), self.lg_ypc, K_CARRIES)
            rows.append({"player_id": pid, "player": last.rusher_player_name, "team": team, "position": "RB",
                         "kind": "rusher", "ypc": ypc, "proj_volume": per_game, "carries": len(gc)})

        # quarterbacks
        att = p[(p["pass"] == 1) & p.passer_player_id.notna()]
        for pid, g in att.groupby("passer_player_id"):
            last = g.sort_values(["season", "week"]).iloc[-1]
            gc = g[g.season == self.season]
            team = last.posteam
            if last.season != self.season or len(gc) < 30:
                continue
            a = g[g.sack.fillna(0) == 0]
            ypa = _shrink(a.yards_gained.mul(a.w).sum(), a.w.sum(), self.lg_ypa, K_ATTEMPTS)
            split = {cov: _shrink(a[a.coverage == cov].yards_gained.mul(a[a.coverage == cov].w).sum(),
                                  a[a.coverage == cov].w.sum(), ypa, 60) for cov in ("man", "zone")}
            pr = g[g.was_pressure.notna()]
            press = {k: _shrink(pr[pr.was_pressure.astype(bool) == v].yards_gained.mul(pr[pr.was_pressure.astype(bool) == v].w).sum(),
                                pr[pr.was_pressure.astype(bool) == v].w.sum(), ypa, 60)
                     for k, v in (("pressured", True), ("clean", False))}
            rows.append({"player_id": pid, "player": last.passer_player_name, "team": team, "position": "QB",
                         "kind": "qb", "ypa": ypa, "ypa_man": split["man"], "ypa_zone": split["zone"],
                         "ypa_pressured": press["pressured"], "ypa_clean": press["clean"],
                         "proj_volume": len(gc[gc.sack.fillna(0) == 0]) / max(team_games.get(team, 1), 1)})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------ one matchup
    def side(self, offense: str, defense: str) -> dict:
        if self.empty or defense not in self.defense.index:
            return {"offense": offense, "defense": defense, "players": pd.DataFrame(), "defense_profile": []}
        d = self.defense.loc[defense]
        n = self.defense.attrs["n"]
        man = d.man_rate
        rows = []
        for p in self.players[self.players.team == offense].itertuples(index=False):
            if p.kind == "receiver":
                rows.append(self._receiver(p, d, defense, man, n))
            elif p.kind == "rusher":
                fac = d.ypc / self.lg_ypc
                base = p.proj_volume * p.ypc
                rows.append(self._row(p, defense, "Rush yards", base, base * fac, fac - 1, [
                    f"{defense} allows {d.ypc:.1f} yards/carry ({fac - 1:+.0%} vs average, rank {int(d.ypc_rank)} of {n}, 1 = toughest)"]))
            elif p.kind == "qb":
                rows.append(self._qb(p, d, defense, man, n))
        table = pd.DataFrame([r for r in rows if r]).sort_values("advantage", ascending=False) if rows else pd.DataFrame()
        return {"offense": offense, "defense": defense, "players": table, "defense_profile": self.profile(defense)}

    def _row(self, p, defense, metric, base, proj, adv, reasons):
        return {"team": p.team, "opponent": defense, "player": p.player, "position": p.position,
                "metric": metric, "baseline": base, "projection": proj, "advantage": adv,
                "grade": _grade(adv), "reasons": reasons}

    def _receiver(self, p, d, defense, man, n):
        lg_pos = self.lg_ypt.get(p.position, self.lg_ypt_all)
        def_fac = d[f"ypt_{p.position}"] / lg_pos
        vs_def = man * p.ypt_man + (1 - man) * p.ypt_zone
        vs_avg = self.lg_man * p.ypt_man + (1 - self.lg_man) * p.ypt_zone
        scheme_fac = vs_def / vs_avg if vs_avg else 1.0
        total = def_fac * scheme_fac
        base = p.proj_volume * p.ypt
        reasons = [f"{defense} allows {d[f'ypt_{p.position}']:.1f} yards/target to {p.position}s "
                   f"({def_fac - 1:+.0%} vs average, rank {int(d[f'ypt_{p.position}_rank'])} of {n}, 1 = toughest)"]
        if self.charted_seasons and abs(scheme_fac - 1) >= 0.02:
            better = "man" if p.ypt_man > p.ypt_zone else "zone"
            reasons.append(f"{defense} plays man on {man:.0%} of dropbacks (league {self.lg_man:.0%}); "
                           f"{p.player} averages {p.ypt_man:.1f} yds/target vs man, {p.ypt_zone:.1f} vs zone "
                           f"(better vs {better}; {p.n_man}/{p.n_zone} targets)")
        for route, info in (p.routes or {}).items():
            allowed = d.route_ypt.get(route) if isinstance(d.route_ypt, dict) else None
            lg = self.lg_route.get(route)
            if allowed and lg and abs(allowed / lg - 1) >= 0.10:
                reasons.append(f"His top route {route.title()} ({info['n']} targets): {defense} allows "
                               f"{allowed:.1f} yds/target on it ({allowed / lg - 1:+.0%} vs average)")
        return self._row(p, defense, "Receiving yards", base, base * total, total - 1, reasons)

    def _qb(self, p, d, defense, man, n):
        def_fac = d.ypa / self.lg_ypa
        scheme = (man * p.ypa_man + (1 - man) * p.ypa_zone) / (self.lg_man * p.ypa_man + (1 - self.lg_man) * p.ypa_zone)
        pr_def, pr_lg = d.pressure_rate, self.lg_pressure
        press = ((pr_def * p.ypa_pressured + (1 - pr_def) * p.ypa_clean) /
                 (pr_lg * p.ypa_pressured + (1 - pr_lg) * p.ypa_clean))
        total = def_fac * scheme * press
        base = p.proj_volume * p.ypa
        reasons = [f"{defense} allows {d.ypa:.1f} yards/attempt ({def_fac - 1:+.0%} vs average, "
                   f"rank {int(d.ypa_rank)} of {n}, 1 = toughest)"]
        if self.charted_seasons:
            reasons.append(f"{defense} pressures on {pr_def:.0%} of dropbacks (league {pr_lg:.0%}); {p.player}: "
                           f"{p.ypa_clean:.1f} yds/att clean, {p.ypa_pressured:.1f} under pressure")
            if abs(scheme - 1) >= 0.02:
                reasons.append(f"vs man {p.ypa_man:.1f} yds/att, vs zone {p.ypa_zone:.1f}; "
                               f"{defense} plays man {man:.0%} (league {self.lg_man:.0%})")
        return self._row(p, defense, "Pass yards", base, base * total, total - 1, reasons)

    def profile(self, defense: str) -> list[str]:
        d = self.defense.loc[defense]
        n = self.defense.attrs["n"]
        lines = []
        if self.charted_seasons and d.charted:
            covs = ", ".join(f"{k.replace('_', ' ').title()} {v:.0%}" for k, v in d.top_coverages.items())
            lines.append(f"Coverage: man {d.man_rate:.0%} / zone {1 - d.man_rate:.0%} (league man {self.lg_man:.0%})"
                         + (f"; most used: {covs}" if covs else ""))
            lines.append(f"Pressure rate {d.pressure_rate:.0%} (league {self.lg_pressure:.0%})")
        lines.append(f"Pass D: {d.ypa:.1f} yds/att allowed (rank {int(d.ypa_rank)} of {n}); "
                     f"Run D: {d.ypc:.1f} yds/carry (rank {int(d.ypc_rank)} of {n})")
        lines.append("Yds/target allowed: " + ", ".join(
            f"{pos} {d[f'ypt_{pos}']:.1f} (rank {int(d[f'ypt_{pos}_rank'])})" for pos in RECEIVER_POS))
        return lines

    def note(self) -> str:
        if self.empty:
            return "No NFL play-by-play yet. Run: python -m scripts.load_matchup_data --league nfl"
        cov = (f"coverage charting for {', '.join(map(str, self.charted_seasons))}"
               if self.charted_seasons else "no coverage charting found (man/zone parts skipped)")
        return f"Play-by-play through the {self.season} season (last season weighted 35%); {cov}."


def game_report(league: str, game_key: str, ctx: NflContext | None = None) -> dict:
    g = db.games_df(league).set_index("game_key").loc[game_key]
    ctx = ctx or NflContext(seasons=default_seasons())
    sides = [ctx.side(g.away_team, g.home_team), ctx.side(g.home_team, g.away_team)] if not ctx.empty else []
    return {"game_key": game_key, "sides": sides, "data_note": ctx.note()}


def default_seasons() -> list[int]:
    from edge.timeutil import et_today
    t = et_today()
    current = t.year if t.month >= 3 else t.year - 1
    return [current - 1, current]
