"""Train, evaluate and (maybe) promote a model, then print the report.

    python -m scripts.train_model --league nba
"""
import argparse
import json

from edge import db
from edge.train import train_league


def pct(x):
    return "n/a" if x is None else f"{x:+.1%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True)
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = ap.parse_args()
    db.init()

    result = train_league(args.league)
    r = result["report"]
    if args.json:
        print(json.dumps(r, indent=2, default=str))
        return

    print(f"\n=== {args.league.upper()} model report ===")
    print(f"Train seasons {r['train_seasons']} ({r['n_train']} games) | "
          f"valid {r['valid_season']} ({r['n_valid']}) | test {r['test_season']} ({r['n_test']})")
    print("\nValidation log loss (lower is better):")
    for kind, c in r["candidates"].items():
        print(f"  {kind:<9} {c['valid']['log_loss']:.4f}")
    print(f"\nWinner: {r['best_kind']} (calibrated)")
    market = "n/a" if r["market_log_loss"] is None else f"{r['market_log_loss']:.4f}"
    print(f"Test log loss  model {r['test']['log_loss']:.4f} | Elo baseline "
          f"{r['test_elo_baseline']['log_loss']:.4f} | market {market}"
          f"  ({r['test_games_with_odds']} test games with odds)")
    bt = r["backtest_test"]
    print(f"\nBacktest on test season at {r['bet_threshold']:.0%} edge threshold "
          f"(chosen on validation): {bt['bets']} bets, ROI {pct(bt['roi'] if bt['bets'] else None)}, "
          f"profit ${bt['profit']:,.0f} at $100/bet")
    print("\nCalibration (test): predicted -> actual win rate")
    for row in r["calibration_test"]:
        print(f"  {row['predicted']:.0%} -> {row['actual']:.0%}  (n={row['n']})")
    print(f"\nSaved {result['version']}  promoted={result['promoted']}")


if __name__ == "__main__":
    main()
