"""Turn a prediction into plain-English reasons.

For LightGBM we use SHAP values; for logistic regression, coefficient x
(standardized) feature value. Both give each feature's push in log-odds,
which we convert to approximate win-probability points for display.
"""
import numpy as np
import pandas as pd


def contributions(bundle, X: pd.DataFrame) -> pd.DataFrame:
    """Per-row, per-feature push toward a HOME win, in log-odds."""
    X = X[bundle.features]
    if bundle.kind == "lgbm":
        import shap
        explainer = shap.TreeExplainer(bundle.base)
        vals = explainer.shap_values(X)
        if isinstance(vals, list):            # older shap: [class0, class1]
            vals = vals[1]
        vals = np.asarray(vals)
        if vals.ndim == 3:                     # (rows, features, classes)
            vals = vals[:, :, 1]
    else:
        imputer, scaler, lr = bundle.base.steps[0][1], bundle.base.steps[1][1], bundle.base.steps[-1][1]
        z = scaler.transform(imputer.transform(X))
        vals = z * lr.coef_[0]
    return pd.DataFrame(vals, columns=bundle.features, index=X.index)


def _tennis_sentence(feat: str, row: pd.Series) -> str | None:
    a, b = row.home_team, row.away_team
    if feat == "elo_diff":
        return f"Overall rating: {a if row.elo_diff > 0 else b} rated {abs(row.elo_diff):.0f} Elo higher"
    if feat == "surface_elo_diff":
        fav = a if row.surface_elo_diff > 0 else b
        return f"{row.get('surface', 'Surface')} court rating: {fav} {abs(row.surface_elo_diff):.0f} Elo higher"
    if feat == "rank_diff":
        if abs(row.rank_diff) < 0.3:
            return None
        fav = a if row.rank_diff > 0 else b
        return f"Ranking: {fav} ranked {2 ** abs(row.rank_diff):.1f}x higher"
    if feat == "form_diff":
        if abs(row.form_diff) < 0.1:
            return None
        fav = a if row.form_diff > 0 else b
        return f"Recent form: {fav} won {abs(row.form_diff):.0%} more of their last 10 matches"
    if feat == "fatigue_diff":
        if row.fatigue_diff == 0:
            return None
        tired = a if row.fatigue_diff > 0 else b
        return f"Workload: {tired} has played {abs(row.fatigue_diff):.0f} more match(es) this week"
    if feat == "rest_diff":
        if abs(row.rest_diff) < 1:
            return None
        fresh = a if row.rest_diff > 0 else b
        return f"Rest: {fresh} has had {abs(row.rest_diff):.0f} more day(s) off"
    return None


def _sentence(feat: str, row: pd.Series, tennis: bool = False) -> str | None:
    if tennis:
        return _tennis_sentence(feat, row)
    home, away = row.home_team, row.away_team
    if feat == "elo_diff":
        fav = home if row.elo_diff > 0 else away
        return f"Team rating: {fav} rated {abs(row.elo_diff):.0f} Elo points higher"
    if feat == "form_diff":
        fav = home if row.form_diff > 0 else away
        return f"Recent form: {fav} winning by {abs(row.form_diff):.1f} more pts/game lately"
    if feat == "season_margin_diff":
        fav = home if row.season_margin_diff > 0 else away
        return f"Season point differential favors {fav} by {abs(row.season_margin_diff):.1f}/game"
    if feat == "rest_diff":
        if abs(row.rest_diff) < 0.5:
            return None
        fav = home if row.rest_diff > 0 else away
        return f"Rest: {fav} has {abs(row.rest_diff):.0f} more day(s) off"
    if feat == "b2b_home":
        return f"{home} on a back-to-back / short week" if row.b2b_home else None
    if feat == "b2b_away":
        return f"{away} on a back-to-back / short week" if row.b2b_away else None
    if feat == "miss_home":
        return f"{home} missing {row.miss_home:.0%} of usual rotation minutes" if row.miss_home >= 0.05 else None
    if feat == "miss_away":
        return f"{away} missing {row.miss_away:.0%} of usual rotation minutes" if row.miss_away >= 0.05 else None
    return None


def reasons_for(bundle, rows: pd.DataFrame, p_home: np.ndarray, sides: list, top: int = 3) -> list[list[str]]:
    """For each game, the top features behind the pick, with their size in
    probability points (+ supports the pick, - works against it)."""
    contrib = contributions(bundle, rows)
    from edge.config import league_config
    tennis = league_config(bundle.league).get("sport") == "tennis"
    out = []
    for (idx, row), p, side in zip(rows.iterrows(), p_home, sides):
        c = contrib.loc[idx]
        slope = p * (1 - p) * 100          # d(prob)/d(log-odds), in points
        sign = 1 if side != "away" else -1
        ranked = c.abs().sort_values(ascending=False).index
        texts = []
        for feat in ranked:
            s = _sentence(feat, row, tennis)
            if s is None:
                continue
            pts = sign * c[feat] * slope
            if abs(pts) < 0.3:
                continue
            texts.append(f"{s} ({pts:+.1f} pts)")
            if len(texts) == top:
                break
        out.append(texts)
    return out
