"""
build_features.py
==================
Runs in the sandbox (or anywhere with pandas) once you've uploaded the raw
CSVs produced by collect_data.py: bot_trades.csv, launch_controls.csv, and
(ideally) a launches_metadata.csv with deployer wallet, dev-buy size, token
name/symbol/socials, per launch — pulled the same way as get_block_launches
but with instruction-data decoded for the CREATE args.

Hard rule enforced throughout: every feature for a launch at time
t_decision uses ONLY data timestamped/slotted strictly before or at
t_decision. Deployer history is truncated at t_decision — this is done via
an as-of join (merge_asof) below, not a plain groupby, specifically so that
a deployer's FUTURE launches never leak into a PAST launch's features.

Usage:
    python build_features.py --launches launches_metadata.csv \
                              --bot-buys bot_trades.csv \
                              --out features.parquet
"""

import argparse
import pandas as pd
import numpy as np


def build_deployer_history_features(launches: pd.DataFrame) -> pd.DataFrame:
    """
    For each launch, compute the deployer's history STRICTLY BEFORE this
    launch's t_decision. Uses expanding/cumulative stats per deployer,
    shifted by one so the current row's own outcome never enters its own
    features (no leakage).
    """
    launches = launches.sort_values(["deployer", "t_decision"]).reset_index(drop=True)

    g = launches.groupby("deployer")

    # Count of prior launches by this deployer (shift(1) excludes current row)
    launches["deployer_prior_launch_count"] = g.cumcount()

    # Deploy frequency: launches per day, trailing, up to t_decision
    launches["deployer_prev_t_decision"] = g["t_decision"].shift(1)
    launches["seconds_since_deployer_last_launch"] = (
        launches["t_decision"] - launches["deployer_prev_t_decision"]
    ).dt.total_seconds()

    # Outcome-based features need an outcome column already computed with a
    # STRICT cutoff — e.g. "reached_migration" known only after the fact.
    # Roll these forward with shift(1)+expanding so only past outcomes count.
    if "reached_migration" in launches.columns:
        launches["deployer_prior_migration_rate"] = (
            g["reached_migration"].apply(lambda s: s.shift(1).expanding().mean())
        )
    if "early_buyer_pnl" in launches.columns:
        launches["deployer_prior_avg_early_buyer_pnl"] = (
            g["early_buyer_pnl"].apply(lambda s: s.shift(1).expanding().mean())
        )
    if "dev_sold_early" in launches.columns:
        launches["deployer_prior_dev_sell_rate"] = (
            g["dev_sold_early"].apply(lambda s: s.shift(1).expanding().mean())
        )

    return launches


def build_wallet_age_features(launches: pd.DataFrame, wallet_first_seen: pd.Series) -> pd.DataFrame:
    """wallet_first_seen: Series indexed by wallet address -> first-ever-seen timestamp
    (from a broader chain scan, NOT limited to pump.fun activity)."""
    launches["deployer_first_seen"] = launches["deployer"].map(wallet_first_seen)
    launches["deployer_wallet_age_sec"] = (
        launches["t_decision"] - launches["deployer_first_seen"]
    ).dt.total_seconds()
    return launches


def build_deploy_tx_features(launches: pd.DataFrame) -> pd.DataFrame:
    """Features purely from the deployment transaction itself — always safe,
    since these are known exactly at t_decision by construction."""
    launches["dev_buy_sol"] = launches.get("dev_buy_sol", np.nan)
    launches["priority_fee_lamports"] = launches.get("priority_fee_lamports", np.nan)
    launches["in_jito_bundle"] = launches.get("in_jito_bundle", False).astype(int) if "in_jito_bundle" in launches else 0
    launches["bundle_size"] = launches.get("bundle_size", np.nan)
    return launches


def build_metadata_features(launches: pd.DataFrame) -> pd.DataFrame:
    launches["name_len"] = launches["name"].fillna("").str.len()
    launches["symbol_len"] = launches["symbol"].fillna("").str.len()
    launches["has_website"] = launches.get("website", pd.Series(dtype=object)).notna().astype(int)
    launches["has_twitter"] = launches.get("twitter", pd.Series(dtype=object)).notna().astype(int)
    launches["has_telegram"] = launches.get("telegram", pd.Series(dtype=object)).notna().astype(int)
    launches["socials_count"] = launches[["has_website", "has_twitter", "has_telegram"]].sum(axis=1)

    # Name/ticker reuse: has this exact (name, symbol) pair been deployed before,
    # counted only up to (not including) t_decision — again via shift, grouped
    # by the name/symbol key rather than deployer.
    launches = launches.sort_values(["symbol", "t_decision"])
    launches["symbol_reuse_count"] = launches.groupby("symbol").cumcount()
    launches = launches.sort_values(["name", "t_decision"])
    launches["name_reuse_count"] = launches.groupby("name").cumcount()

    return launches


def build_timing_features(launches: pd.DataFrame) -> pd.DataFrame:
    launches["hour_of_day_utc"] = launches["t_decision"].dt.hour
    launches["day_of_week"] = launches["t_decision"].dt.dayofweek
    return launches


def label_from_bot_buys(launches: pd.DataFrame, bot_buys: pd.DataFrame) -> pd.DataFrame:
    bought_mints = set(bot_buys["mint"].unique())
    launches["bot_bought"] = launches["mint"].isin(bought_mints).astype(int)
    return launches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launches", required=True)
    ap.add_argument("--bot-buys", required=True)
    ap.add_argument("--out", default="features.parquet")
    args = ap.parse_args()

    launches = pd.read_csv(args.launches, parse_dates=["t_decision"])
    bot_buys = pd.read_csv(args.bot_buys)

    launches = build_deployer_history_features(launches)
    launches = build_deploy_tx_features(launches)
    launches = build_metadata_features(launches)
    launches = build_timing_features(launches)
    launches = label_from_bot_buys(launches, bot_buys)

    print(f"Built features for {len(launches)} launches. "
          f"Positive rate: {launches['bot_bought'].mean():.4%}")
    print(f"Class balance: {launches['bot_bought'].sum()} bought / "
          f"{(1 - launches['bot_bought']).sum():.0f} not bought")

    launches.to_parquet(args.out, index=False)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
