# Solana Zero-Block Sniper — Reverse Engineering & Replica Strategy

Reverse-engineers the token-selection logic of a pump.fun zero-block sniper
bot (`5brv79eFZ2rGprXNvqgVJBkBptkkw8GJX1XydJyZLyAr`) and builds a replica
scoring strategy, backtested against the bot's real performance.

## Pipeline

```
collect_data.py   -->  build_features.py  -->  train_model.py  -->  backtest.py
(raw chain data)       (t_decision-safe        (time-split model,     (replica strategy,
                         feature table)          top-10 importances)    ROI/drawdown, overlap)
```

## Setup

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt --break-system-packages   # or use a venv
```

Set your Helius API key (free tier at https://helius.dev) in `collect_data.py`,
or export it and adjust the script to read from env:

```bash
export HELIUS_API_KEY=your_key_here
```

## Step 1 — Collect raw data (network required, run locally or in a
GitHub Codespace; several hours for the full window)

```bash
python collect_data.py
```

Resumable — re-run to continue from the last checkpoint in `pumpbot_data/`.
Step 0 of the script prints the bot's true first/last trade timestamps;
use that to confirm/adjust the collection window before letting it run long.

Outputs: `pumpbot_data/bot_trades.csv`, `pumpbot_data/launch_controls.csv`.

**Known TODOs before trusting this at full scale** (flagged inline in the
script): confirm the pump.fun CREATE-instruction discriminator bytes against
one known launch transaction, and verify the Helius Enhanced API response
shape (`type`/`instructions` fields) against one live decoded record — both
can drift as pump.fun/Helius ship changes.

## Step 2 — Feature engineering (t_decision-truncated)

```bash
python build_features.py \
  --launches pumpbot_data/launches_metadata.csv \
  --bot-buys pumpbot_data/bot_trades.csv \
  --out features.parquet
```

All deployer-history features are computed via an as-of (shift + expanding)
join per deployer — never a plain aggregate — so no launch's features can
see that deployer's *future* launches. This is the mechanism enforcing the
brief's hard constraint.

## Step 3 — Train and evaluate

```bash
python train_model.py --features features.parquet --out model_report/
```

Time-based (not random) train/test split. Outputs `metrics.json`
(precision/recall/F1/PR-AUC at an F1-optimal threshold), `top10_features.png`,
`pr_curve.png`, and `test_predictions.parquet` for the backtest step.

## Step 4 — Backtest the replica strategy

```bash
python backtest.py \
  --predictions model_report/test_predictions.parquet \
  --outcomes pumpbot_data/token_outcomes.csv \
  --bot-trades pumpbot_data/bot_trades.csv \
  --threshold <value from metrics.json> \
  --out backtest_report/
```

`token_outcomes.csv` (realized ROI per mint, used only for post-hoc grading,
never as a training feature) needs to be built separately from post-deploy
price data — not included in this repo; see the "Outcomes data" note below.

Outputs per-strategy stats (hit rate, avg ROI, max drawdown), equity curve
plots at 0/1/2 slot entry delay, and overlap precision/recall vs. the bot's
actual buys.

## Outcomes data note

This repo intentionally does not ship a price-outcome table — building one
requires pulling post-deployment trade history per mint (bonding-curve price
path or migration outcome), which is a separate, much larger pull than the
bot's own trade history. A minimal version only needs outcomes for mints the
model actually scores highly plus the bot's own buys, not the full universe.

## Reproducing on Kaggle

The public Kaggle Notebook (linked in the Writeup) runs Steps 2-4 directly
against pre-collected CSVs uploaded as a Kaggle Dataset, since Step 1 needs
long-running network access that Kaggle's execution environment doesn't
reliably provide.
