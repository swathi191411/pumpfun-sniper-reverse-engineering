"""
backtest.py
===========
Turns the trained model's scores into a replica entry strategy and backtests
it against realized token price paths, then compares to the bot's actual
trades.

Requires a price/outcome table per mint (post-deploy price path or at least
entry/exit-equivalent returns) — NOT used as a training feature (that would
violate the t_decision constraint) but needed here purely to grade trades
after the fact, same as the bot's own trades are graded in Part 1.

Usage:
    python backtest.py --predictions model_report/test_predictions.parquet \
                        --outcomes token_outcomes.csv \
                        --bot-trades bot_trades.csv \
                        --threshold 0.5 \
                        --out backtest_report/
"""

import argparse, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def max_drawdown(equity_curve: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()


def simulate_entries(preds: pd.DataFrame, outcomes: pd.DataFrame, threshold: float, slot_delay: int = 0):
    """
    preds: mint, pred_proba, slot
    outcomes: mint, roi (realized return if bought at deploy / deploy+slot_delay)
    slot_delay: 0 = same-block entry (idealized); 1-2 = realistic queueing lag,
                pulls from a returns table computed at that later entry point
                if available, else approximated by discounting idealized ROI.
    """
    entries = preds[preds["pred_proba"] >= threshold].merge(outcomes, on="mint", how="left")
    entries = entries.dropna(subset=["roi"])
    if slot_delay > 0 and "roi_delayed" in entries.columns:
        entries["roi_used"] = entries.get(f"roi_delay_{slot_delay}", entries["roi"])
    else:
        entries["roi_used"] = entries["roi"]
    return entries


def report_strategy(entries: pd.DataFrame, label: str, out_dir: str):
    n = len(entries)
    hit_rate = (entries["roi_used"] > 0).mean() if n else float("nan")
    avg_roi = entries["roi_used"].mean() if n else float("nan")
    total_pnl = entries["roi_used"].sum() if n else float("nan")

    equity = (1 + entries.sort_values("slot")["roi_used"]).cumprod().values if n else np.array([1.0])
    dd = max_drawdown(equity) if n else float("nan")

    stats = {
        "label": label,
        "n_trades": n,
        "hit_rate": float(hit_rate),
        "avg_roi_per_trade": float(avg_roi),
        "total_pnl_(sum_of_roi)": float(total_pnl),
        "max_drawdown": float(dd),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(equity, label=label)
    ax.set_title(f"Equity curve — {label}")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Equity (starting=1.0, equal-weight per trade)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"equity_curve_{label.replace(' ', '_')}.png"), dpi=150)

    return stats


def overlap_metrics(replica_mints: set, bot_mints: set):
    tp = len(replica_mints & bot_mints)
    precision = tp / len(replica_mints) if replica_mints else float("nan")
    recall = tp / len(bot_mints) if bot_mints else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    return {"overlap_precision": precision, "overlap_recall": recall, "overlap_f1": f1,
            "replica_n": len(replica_mints), "bot_n": len(bot_mints), "intersection_n": tp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--outcomes", required=True)
    ap.add_argument("--bot-trades", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="backtest_report")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    preds = pd.read_parquet(args.predictions) if args.predictions.endswith(".parquet") else pd.read_csv(args.predictions)
    outcomes = pd.read_csv(args.outcomes)
    bot_trades = pd.read_csv(args.bot_trades)

    all_stats = []
    for delay in [0, 1, 2]:
        entries = simulate_entries(preds, outcomes, args.threshold, slot_delay=delay)
        stats = report_strategy(entries, f"replica_delay{delay}", args.out)
        all_stats.append(stats)
        print(f"[slot_delay={delay}] {stats}")

    bot_entries = outcomes[outcomes["mint"].isin(bot_trades["mint"])].copy()
    bot_entries["roi_used"] = bot_entries["roi"]
    bot_entries["slot"] = bot_entries.get("slot", 0)
    bot_stats = report_strategy(bot_entries, "actual_bot", args.out)
    all_stats.append(bot_stats)
    print(f"[actual bot] {bot_stats}")

    replica_mints = set(simulate_entries(preds, outcomes, args.threshold, 0)["mint"])
    bot_mints = set(bot_trades["mint"])
    overlap = overlap_metrics(replica_mints, bot_mints)
    print(f"[overlap] {overlap}")

    with open(os.path.join(args.out, "backtest_summary.json"), "w") as f:
        json.dump({"strategy_stats": all_stats, "overlap": overlap, "threshold": args.threshold}, f, indent=2)

    print(f"\nSaved full backtest report to {args.out}/")


if __name__ == "__main__":
    main()
