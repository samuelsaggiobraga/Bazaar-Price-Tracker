# Model performance metrics

Out-of-sample results recorded by `Backend/backtest.py`. One JSON per item,
`<ITEM>_holdout.json`, containing the config, the data split, model metrics at
several confidence thresholds, and a trade-level backtest.

Regenerate with a training run that holds out the tail of the series; the JSON
records `train_span` / `test_span` so any result can be tied back to its window.

---

## How to read these numbers

**Quote the fixed-stake figures, not the compounded ones.** Compounding full
capital across hundreds of trades implies order-book depth that does not exist —
it yields numbers like 10^23%. `stake_1M` / `stake_10M` re-use a fixed stake each
trade, which is how a flipper actually operates.

**P&L is decomposed, and the split matters more than the total.** A Bazaar flip
earns two separate things:

| component | what it is | credit to |
|---|---|---|
| `spread_capture` | the bid/ask gap already quoted at entry (median 12.8% on CONTROL_SWITCH) | the order book — anyone whose limit orders fill |
| `directional` | how far `buy_price` moved while held | **the model** — this is the only part it predicts |

Reporting the total alone credits the model for the spread. It didn't earn it.

**`directional_edge_vs_random_pp` is the primary metric.** Same rules, same fill
assumptions, same period, same number of entries — only the *timing* differs.
That difference is the model's contribution and nothing else.

**Fill assumptions are optimistic and stated.** Entry assumes a buy order fills
at the bid; exit assumes a sell offer fills at the ask; 1.25% tax on every sale.
Whether those orders actually fill is the hard part of flipping and cannot be
verified from price history. Treat the spread component as an upper bound.

---

## CONTROL_SWITCH — 2026-06-05 → 2026-08-04

Trained on 511,511 rows (2023-04-18 → 2026-06-05), tested on 28,931 rows.
The test window is data the published HuggingFace snapshot never contained —
it was fetched live and the model had never seen any of it.

Config: `LABEL_MODE=median`, `OBJECTIVE=binary`, 30 Optuna trials, confidence ≥ 0.80.

### Model

| metric | value |
|---|---|
| ROC AUC | **0.9091** |
| shuffled-label control | **0.4596** (chance — the signal is real) |
| precision | 0.962 |
| positive base rate | 0.687 |

Precision reads better than it is: 68.7% of test bars were already positive
because the item trended up. The lift over the base rate is ~1.4×.

### Backtest — 265 trades, 4.4/day, 110 min average hold, 33.7% exposure

Mean per trade:

| component | per trade |
|---|---|
| spread capture | +19.92% |
| directional | +4.86% |
| **total** | **+24.78%** |
| random-entry control (directional) | +1.49% |
| **model edge over random** | **+3.36 pp** |

At a fixed 1,000,000 coin stake, over the two months:

| | coins |
|---|---|
| total profit | 65,671,268 |
| — of which spread capture | 52,800,618 |
| — **attributable to the model** | **8,914,246** |
| worst single trade | −104,118 |

So roughly **80% of the profit is spread capture** available to any filled
order, and ~14% is the model's timing edge.

### What to be careful about

- **Directional win rate is 34%** even though total win rate is 98.5% — the
  spread makes almost every trade look like a winner. The model is wrong more
  often than right; it wins on the size of its winners.
- **Directional max drawdown is −84%.** The strategy is not smooth.
- **179 days of gaps** (52 holes > 6h) exist in the underlying history, partly
  from API chunks that returned 400. Gaps distort forward windows near them.
- **One item, one contiguous window.** A walk-forward across several origins is
  the missing validation before any of this should size real positions.
- Exit mix at conf ≥ 0.80: 119 stop-loss, 80 timeout, 66 take-profit.
