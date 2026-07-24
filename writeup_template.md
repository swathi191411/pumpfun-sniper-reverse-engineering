# Reverse-Engineering a Zero-Block Pump.fun Sniper
## [Subtitle: one-sentence summary of your final replica strategy's edge]

<!-- Word budget: 3000 words max. This scaffold is intentionally terse — fill
in numbers/plots from your actual run, don't pad with prose. -->

## Part 1 — Behavioral analysis of the competitor

- Total tokens bought: `N` over `[date range]`
- Entry size: mean `X SOL`, median `Y SOL`, std `Z SOL` — [insert histogram]
- Latency: `P%` zero-block entries; median latency `X ms` / `Y slots`; block
  position — `A%` same-bundle-as-deploy, `B%` next tx, `C%` later same block
- Hold time distribution: median `X min`, [insert histogram]
- Exit structure: `X%` single-tx exit, `Y%` partial/staged exits (avg `N`
  sell txs/token), `Z%` involve a burn
- Hit rate: `X%`; avg win `+Y%`, avg loss `-Z%`; [insert P&L distribution plot]

## Part 2 — Reverse-engineering the bot's features

**Constraint compliance**: all features computed strictly before/at
`t_decision`; deployer history features use an as-of (shift+expanding) join,
never a plain aggregate — see `build_features.py::build_deployer_history_features`.

**Model**: LightGBM classifier, time-based train/test split at `[cutoff date]`,
`scale_pos_weight` for the ~16K/5M imbalance rather than resampling (preserves
calibration for backtest thresholding).

**Held-out test performance** (bot-bought class):
| Metric | Value |
|---|---|
| PR-AUC | `X` |
| Precision @ threshold `T` | `X` |
| Recall @ threshold `T` | `X` |
| F1 @ threshold `T` | `X` |

[insert PR curve plot]

**Top-10 features by importance:**
[insert top10_features.png + table]

**Hypothesized rules (plain language):**
1. `[e.g. "Deployer must have ≥N prior launches with migration rate >X%"]`
2. `[e.g. "Dev-buy must fall in range [a,b] SOL"]`
3. ...

## Part 3 — Replica strategy and backtest

**Scoring rule**: model probability ≥ `T` → enter; threshold chosen by
`[F1-maximizing / precision-target / other]` on validation.

**Entry feasibility**: [discuss what's realistically achievable — bundle
inclusion, priority fee competition — and results at 0/1/2 slot delay]

| Slots delay | Trades | Hit rate | Avg ROI/trade | Max DD | Total P&L |
|---|---|---|---|---|---|
| 0 | | | | | |
| 1 | | | | | |
| 2 | | | | | |

**Comparison vs. the actual bot:**
| | Replica | Bot |
|---|---|---|
| N trades | | |
| Hit rate | | |
| Avg ROI/trade | | |
| Max drawdown | | |

Token-selection overlap: precision `X%`, recall `Y%`, F1 `Z%` (replica vs.
bot's actual buys).

[insert side-by-side equity curves]

**Ideas for improvement:**
- `[e.g. wallet-clustering to catch deployer alt-accounts]`
- `[e.g. incorporate bundle/Jito tip size as a competitiveness signal]`
- ...
