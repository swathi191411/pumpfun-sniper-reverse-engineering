"""
train_model.py
===============
Time-based train/test split (never random split — this is a temporal
prediction problem and random splits leak future deployer-history features
into the past). Handles severe class imbalance (~16K positive / ~5M
negative) via class weighting rather than naive resampling, so probability
calibration stays meaningful for the backtest's threshold choice later.

Usage:
    python train_model.py --features features.parquet --out model_report/
"""

import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURE_COLS = [
    "deployer_prior_launch_count",
    "seconds_since_deployer_last_launch",
    "deployer_prior_migration_rate",
    "deployer_prior_avg_early_buyer_pnl",
    "deployer_prior_dev_sell_rate",
    "deployer_wallet_age_sec",
    "dev_buy_sol",
    "priority_fee_lamports",
    "in_jito_bundle",
    "bundle_size",
    "name_len",
    "symbol_len",
    "socials_count",
    "has_website",
    "has_twitter",
    "has_telegram",
    "symbol_reuse_count",
    "name_reuse_count",
    "hour_of_day_utc",
    "day_of_week",
]


def time_based_split(df, time_col="t_decision", test_frac=0.2):
    df = df.sort_values(time_col)
    cutoff_idx = int(len(df) * (1 - test_frac))
    cutoff_time = df.iloc[cutoff_idx][time_col]
    train = df[df[time_col] < cutoff_time]
    test = df[df[time_col] >= cutoff_time]
    return train, test, cutoff_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="model_report")
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_parquet(args.features)
    cols = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(cols)
    if missing:
        print(f"WARNING: missing feature columns, skipping: {missing}")

    train, test, cutoff = time_based_split(df, test_frac=args.test_frac)
    print(f"Time-based split at {cutoff}: {len(train)} train / {len(test)} test")
    print(f"Train positive rate: {train['bot_bought'].mean():.4%}, "
          f"Test positive rate: {test['bot_bought'].mean():.4%}")

    X_train, y_train = train[cols], train["bot_bought"]
    X_test, y_test = test[cols], test["bot_bought"]

    # Class imbalance: weight positives up rather than undersample negatives,
    # to preserve the true prior for probability calibration.
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        scale_pos_weight=pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    ap_score = average_precision_score(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)

    # Choose operating point by maximizing F1 on the PR curve (report a few
    # thresholds since imbalance makes a single point misleading).
    prec, rec, thresh = precision_recall_curve(y_test, probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = np.nanargmax(f1s[:-1]) if len(f1s) > 1 else 0
    best_thresh = thresh[best_idx] if len(thresh) > best_idx else 0.5

    y_pred = (probs >= best_thresh).astype(int)
    report = {
        "cutoff_time": str(cutoff),
        "n_train": len(train),
        "n_test": len(test),
        "test_positive_rate": float(y_test.mean()),
        "pr_auc": float(ap_score),
        "roc_auc": float(roc_auc),
        "chosen_threshold": float(best_thresh),
        "precision_at_threshold": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_at_threshold": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    # Top-10 feature importances
    importances = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
    top10 = importances.head(10)
    top10.to_csv(os.path.join(args.out, "top10_features.csv"))
    print("\nTop-10 features by importance:")
    print(top10)

    fig, ax = plt.subplots(figsize=(8, 5))
    top10[::-1].plot.barh(ax=ax)
    ax.set_xlabel("LightGBM gain importance")
    ax.set_title("Top-10 features — bot buy-decision model")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "top10_features.png"), dpi=150)

    # PR curve plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rec, prec, label=f"PR-AUC={ap_score:.4f}")
    ax.axhline(y_test.mean(), color="gray", linestyle="--", label="baseline (positive rate)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (test, time-based split)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "pr_curve.png"), dpi=150)

    model.booster_.save_model(os.path.join(args.out, "model.txt"))
    test.assign(pred_proba=probs).to_parquet(os.path.join(args.out, "test_predictions.parquet"))
    print(f"\nSaved model, metrics, plots, and test predictions to {args.out}/")


if __name__ == "__main__":
    main()
