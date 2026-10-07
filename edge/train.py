"""Train, compare, calibrate and save win-probability models.

Protocol (the part to explain in interviews):
  1. Time-based split by season: train on old seasons, pick the model on the
     next season (validation), and score the most recent season once (test).
     Random shuffling would leak the future into training and inflate results.
  2. Candidates: Elo-only baseline, logistic regression, LightGBM.
  3. The validation winner gets a calibration layer (Platt/sigmoid) so that
     "60%" really means about 60%. The edge math depends on that.
  4. Benchmarks on the test season: the Elo baseline (easy to beat) and the
     closing-line market (hard to beat).
  5. Production model: refit on everything except the most recent games,
     calibrate on those, save with a version, promote only if it isn't worse.
"""
import json
import logging
import warnings
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select, update

from edge import db
from edge.backtest import choose_threshold, market_log_loss, simulate
from edge.config import MODEL_DIR, league_config
from edge.features import build_features, feature_list
from edge.market import closing_lines
from edge.timeutil import utcnow_iso

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def make_model(kind: str):
    if kind in ("elo_only", "logreg"):
        return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                  min_child_samples=50, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
    raise ValueError(kind)


@dataclass
class ModelBundle:
    """Everything needed to make and explain a prediction, saved as one file."""
    league: str
    kind: str
    features: list[str]
    base: object                 # the fitted model (used for explanations)
    calibrated: object           # base + calibration layer (used for probabilities)
    version: str = ""
    bet_threshold: float = 0.04
    report: dict = field(default_factory=dict)
    background: pd.DataFrame | None = None   # sample of training rows for explanations

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.calibrated.predict_proba(df[self.features])[:, 1]


# --------------------------------------------------------------- data
def load_training_frame(league: str) -> pd.DataFrame:
    cfg = league_config(league)
    games = db.games_df(league)
    if games.empty:
        return games
    minutes = db.minutes_df(league) if cfg.get("use_minutes") else None
    df = build_features(games, cfg, minutes=minutes)
    df = df[(df.season_type != "preseason") & df.home_win.notna()].copy()
    close = closing_lines(db.odds_df(league), games)
    return df.merge(close, on="game_key", how="left").sort_values("start_time").reset_index(drop=True)


def season_split(df: pd.DataFrame):
    """Returns (train, valid, test, info). Only seasons with a reasonable number
    of games are used for valid/test (a season that just started is too small)."""
    counts = df.groupby("season").size()
    big = counts[counts >= 0.5 * counts.max()].index.sort_values()
    if len(big) < 3:
        raise ValueError(
            f"Need at least 3 full seasons to train and evaluate (have {list(counts.index)}). "
            "Run scripts/backfill.py for more seasons.")
    valid_s, test_s = big[-2], big[-1]
    train = df[df.season < valid_s]
    valid = df[df.season == valid_s]
    test = df[df.season == test_s]
    return train, valid, test, {"valid_season": int(valid_s), "test_season": int(test_s),
                                "train_seasons": sorted(int(s) for s in train.season.unique())}


def _scores(y, p) -> dict:
    return {"log_loss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p))}


def calibrate(model, X, y):
    cal = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    cal.fit(X, y)
    return cal


def calibration_table(y, p, bins: int = 10) -> list[dict]:
    d = pd.DataFrame({"y": y, "p": p})
    d["bin"] = pd.cut(d.p, np.linspace(0, 1, bins + 1), include_lowest=True)
    t = d.groupby("bin", observed=True).agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    return [{"predicted": float(r.predicted), "actual": float(r.actual), "n": int(r.n)}
            for r in t.itertuples() if r.n > 0]


# --------------------------------------------------------------- evaluate
def evaluate(league: str, df: pd.DataFrame | None = None) -> dict:
    """Honest model comparison. Returns a report dict (also logged to MLflow if installed)."""
    cfg = league_config(league)
    df = load_training_frame(league) if df is None else df
    train, valid, test, info = season_split(df)
    report = {"league": league, **info, "n_train": len(train), "n_valid": len(valid),
              "n_test": len(test), "candidates": {}}

    fitted = {}
    for kind in cfg.get("model_kinds", ["elo_only", "logreg"]):
        feats = feature_list(cfg, kind)
        model = make_model(kind).fit(train[feats], train.home_win)
        p_valid = model.predict_proba(valid[feats])[:, 1]
        report["candidates"][kind] = {"features": feats, "valid": _scores(valid.home_win, p_valid)}
        fitted[kind] = (model, feats)
        _mlflow_log(f"{league}-{kind}", {"kind": kind, "league": league}, report["candidates"][kind]["valid"])

    best_kind = min(report["candidates"], key=lambda k: report["candidates"][k]["valid"]["log_loss"])
    model, feats = fitted[best_kind]
    cal = calibrate(model, valid[feats], valid.home_win)
    p_valid = cal.predict_proba(valid[feats])[:, 1]
    p_test = cal.predict_proba(test[feats])[:, 1]

    # Elo baseline on the same test season, for the "did ML help?" question
    elo_model, elo_feats = fitted.get("elo_only") or (make_model("elo_only").fit(train[["elo_diff"]], train.home_win), ["elo_diff"])
    p_test_elo = elo_model.predict_proba(test[elo_feats])[:, 1]

    # Require a meaningful number of bets before trusting a threshold's ROI (5% of games, min 30).
    min_bets = max(30, len(valid) // 20)
    threshold, threshold_table = choose_threshold(valid, p_valid, min_bets=min_bets,
                                                  default=cfg.get("bet_edge", 0.04))
    _, bt = simulate(test, p_test, threshold)

    report.update({
        "best_kind": best_kind,
        "features": feats,
        "test": _scores(test.home_win, p_test),
        "test_elo_baseline": _scores(test.home_win, p_test_elo),
        "market_log_loss": market_log_loss(test),
        "test_games_with_odds": int(test.close_home_ml.notna().sum()) if "close_home_ml" in test else 0,
        "bet_threshold": threshold,
        "threshold_table_valid": threshold_table,
        # Did ANY threshold make money on the validation season? If not, the app
        # caps every pick at Lean: the model hasn't shown it can beat the market.
        "validation_profitable": any(t["roi"] > 0 and t["bets"] >= min_bets for t in threshold_table),
        "backtest_test": bt,
        "calibration_test": calibration_table(test.home_win, p_test),
    })
    _mlflow_log(f"{league}-{best_kind}-calibrated", {"kind": best_kind, "league": league, "threshold": threshold},
                {"test_log_loss": report["test"]["log_loss"], "test_brier": report["test"]["brier"],
                 "backtest_roi": bt["roi"], "backtest_bets": bt["bets"]})
    return report


# --------------------------------------------------------------- production
def fit_production(league: str, report: dict, df: pd.DataFrame | None = None) -> ModelBundle:
    """Refit the chosen model on (almost) all data; calibrate on the newest games."""
    cfg = league_config(league)
    df = load_training_frame(league) if df is None else df
    kind, feats = report["best_kind"], report["features"]
    n_cal = min(cfg.get("calibration_games", 600), len(df) // 4)
    fit_part, cal_part = df.iloc[:-n_cal], df.iloc[-n_cal:]
    base = make_model(kind).fit(fit_part[feats], fit_part.home_win)
    cal = calibrate(base, cal_part[feats], cal_part.home_win)
    version = f"{league}-{kind}-{utcnow_iso().replace(':', '').replace('-', '')}"
    bundle = ModelBundle(
        league=league, kind=kind, features=feats, base=base, calibrated=cal,
        version=version, bet_threshold=report.get("bet_threshold", cfg.get("bet_edge", 0.04)),
        report={**report, "data_cutoff": str(df.start_time.max()), "n_production": len(df)},
        background=fit_part[feats].sample(min(200, len(fit_part)), random_state=0),
    )
    return bundle


def save_and_maybe_promote(bundle: ModelBundle, tolerance: float = 0.002) -> bool:
    """Save the model; make it active if there is no active model, if the test
    season moved on, or if it scores at least as well as the active one."""
    db.init()
    path = MODEL_DIR / f"{bundle.version}.joblib"
    joblib.dump(bundle, path)
    r = bundle.report
    active = active_row(bundle.league)
    promote = (active is None
               or json.loads(active["report"]).get("test_season") != r.get("test_season")
               or r["test"]["log_loss"] <= active["test_log_loss"] + tolerance)
    with db.get_engine().begin() as conn:
        if promote:
            conn.execute(update(db.model_versions).where(db.model_versions.c.league == bundle.league)
                         .values(is_active=0))
        conn.execute(db.model_versions.insert().values(
            version=bundle.version, league=bundle.league, kind=bundle.kind, trained_at=utcnow_iso(),
            data_cutoff=r.get("data_cutoff"), features=json.dumps(bundle.features),
            valid_log_loss=r["candidates"][bundle.kind]["valid"]["log_loss"],
            test_log_loss=r["test"]["log_loss"], test_brier=r["test"]["brier"],
            market_log_loss=r.get("market_log_loss"), bet_threshold=bundle.bet_threshold,
            backtest_roi=r["backtest_test"]["roi"], backtest_bets=r["backtest_test"]["bets"],
            n_train=r.get("n_production"), path=str(path), is_active=int(promote),
            report=json.dumps(r, default=str)))
    return promote


def active_row(league: str) -> dict | None:
    mv = db.model_versions
    with db.get_engine().connect() as conn:
        row = conn.execute(select(mv).where((mv.c.league == league) & (mv.c.is_active == 1))).mappings().first()
    return dict(row) if row else None


def load_active(league: str) -> ModelBundle | None:
    row = active_row(league)
    if row is None:
        return None
    return joblib.load(row["path"])


def train_league(league: str) -> dict:
    """Full weekly retrain: evaluate -> fit production -> save/promote."""
    df = load_training_frame(league)
    report = evaluate(league, df)
    bundle = fit_production(league, report, df)
    promoted = save_and_maybe_promote(bundle)
    return {"version": bundle.version, "promoted": promoted, "report": report}


# --------------------------------------------------------------- mlflow (optional)
def _mlflow_log(run_name: str, params: dict, metrics: dict) -> None:
    try:
        import mlflow
    except ImportError:
        return
    try:
        from edge.config import DATA_DIR
        # View with:  mlflow ui --backend-store-uri sqlite:///data/mlflow.db
        mlflow.set_tracking_uri(f"sqlite:///{(DATA_DIR / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("edge-finder")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
    except Exception as e:  # noqa: BLE001 - tracking must never break training
        log.warning("MLflow logging skipped: %s", e)
