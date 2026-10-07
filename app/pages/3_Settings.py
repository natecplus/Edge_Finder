"""League switches, thresholds, data health, and manual refresh buttons."""
import streamlit as st

from common import setup

from edge import db
from edge.config import league_config
from edge.timeutil import parse_utc, utcnow

league = setup("Settings")
st.title("Settings")

cfg = league_config(league)
overrides = db.get_setting(f"{league}.overrides", {}) or {}

st.subheader(f"{league.upper()}")
enabled = st.toggle("League enabled", overrides.get("enabled", True))
custom = st.toggle("Override the model's tuned thresholds", "bet_edge" in overrides,
                   help="By default the Bet threshold is the one chosen on the validation season.")
bet = st.slider("Bet when edge is at least (pts)", 1.0, 20.0,
                float(overrides.get("bet_edge", cfg["bet_edge"])) * 100, 0.5, disabled=not custom)
lean = st.slider("Lean when edge is at least (pts)", 0.5, 10.0,
                 float(overrides.get("lean_edge", cfg["lean_edge"])) * 100, 0.5, disabled=not custom)
if st.button("Save settings", type="primary"):
    new = {"enabled": enabled}
    if custom:
        new.update(bet_edge=bet / 100, lean_edge=min(lean, bet) / 100)
    db.set_setting(f"{league}.overrides", new)
    st.success("Saved. New thresholds apply from the next refresh.")

st.divider()
st.subheader("Data health")
runs = db.read_sql("SELECT source, task, ran_at, ok, rows, error FROM source_runs ORDER BY id")
if runs.empty:
    st.write("No data fetched yet.")
else:
    last = runs.groupby(["source", "task"]).agg(
        last_run=("ran_at", "max"),
        last_success=("ran_at", lambda s: s[runs.loc[s.index, "ok"] == 1].max()),
        runs=("ok", "size"), failures=("ok", lambda s: int((s == 0).sum()))).reset_index()
    now = utcnow()

    def age(ts):
        if not isinstance(ts, str):
            return None
        return round((now - parse_utc(ts)).total_seconds() / 3600, 1)

    last["hours_since_success"] = last.last_success.map(age)
    stale = last.hours_since_success.isna() | (last.hours_since_success > 24)
    if stale.any():
        st.error("Some sources have not succeeded in over 24 hours. Check the errors below; "
                 "a site may have changed its format.")
    st.dataframe(last.style.apply(
        lambda r: ["background-color: #ffebe9" if stale[r.name] else "" for _ in r], axis=1),
        hide_index=True, width="stretch")
    errors = runs[runs.ok == 0].tail(10)
    if not errors.empty:
        with st.expander("Latest errors"):
            st.dataframe(errors, hide_index=True, width="stretch")

st.divider()
st.subheader("Run now")
c1, c2 = st.columns(2)
if c1.button("Refresh today's games + picks"):
    from edge import jobs
    with st.spinner("Fetching odds and injuries..."):
        jobs.game_day(league)
    st.cache_data.clear()
    st.success("Done.")
if c2.button("Retrain model"):
    from edge import train
    with st.spinner("Training (a minute or two)..."):
        result = train.train_league(league)
    st.success(f"Trained {result['version']} · promoted: {result['promoted']}")

models = db.read_sql("SELECT version, kind, trained_at, test_log_loss, market_log_loss, backtest_roi, "
                     "backtest_bets, is_active FROM model_versions WHERE league = :lg ORDER BY trained_at DESC",
                     lg=league)
if not models.empty:
    st.subheader("Model versions")
    st.dataframe(models, hide_index=True, width="stretch")
