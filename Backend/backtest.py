"""Trade-level backtester and metrics recorder.

AUC does not tell you what a strategy earns, so this walks the held-out period
bar by bar, opens and closes real positions, and records P&L to metrics/.

Why P&L is decomposed
---------------------
A Bazaar flip earns two different things, and conflating them produces
nonsense. Entering at ``sell`` (your buy order is hit) and exiting at ``buy``
(your sell offer is hit) banks the quoted spread -- on CONTROL_SWITCH that is a
median 12.8%, which dwarfs any 2% take-profit and makes every trade a winner the
instant it opens. That is not skill, it is the spread, and it is only realisable
if BOTH limit orders actually fill -- the hard part of flipping, and something
price history alone cannot verify.

So each trade is split exactly the way ``entry_label`` is built:

    spread_capture = (buy[i]*(1-tax) - sell[i]) / sell[i]
    directional    = (buy[j] - buy[i]) * (1-tax) / sell[i]
    total          = spread_capture + directional

``directional`` is the only component the model predicts, and the only one that
can be credited to it. ``spread_capture`` is what the order book was already
offering at entry, available to anyone whose orders fill. Reporting the total
alone would credit the model for the spread.

Fill assumptions are stated, not hidden:
  - entry at ``sell[i]``  assumes a buy order fills at the bid
  - exit  at ``buy[j]``   assumes a sell offer fills at the ask
  - 1.25% Bazaar tax on every sale
  - ``min_hold_minutes`` prevents the fiction of a round trip inside one bar
  - one position at a time; ``cooldown_minutes`` throttles re-entry

Compounding assumes full capital per trade, which real order-book depth will not
support. Mean return per trade is the honest headline; compounded is reported
with a ruin flag.
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TAX = 0.0125
METRICS_DIR = os.path.join(project_root, "metrics")


def simulate_trades(
    df,
    signal,
    take_profit=0.02,
    stop_loss=0.02,
    max_hold_minutes=180,
    min_hold_minutes=30,
    cooldown_minutes=60,
    tax=TAX,
):
    """One position at a time; TP/SL evaluated on the DIRECTIONAL component.

    TP/SL are applied to ``directional`` rather than ``total`` because the
    spread is booked at entry and is not something you can take profit on.
    """
    ts = pd.to_datetime(df["timestamp"]).values.astype("datetime64[s]").astype(np.int64)
    buy = df["buy_price"].to_numpy(dtype=float)
    sell = df["sell_price"].to_numpy(dtype=float)
    signal = np.asarray(signal, dtype=bool)
    n = len(df)
    max_hold, min_hold = max_hold_minutes * 60, min_hold_minutes * 60
    cooldown = cooldown_minutes * 60

    trades = []
    in_pos = False
    entry_i = 0
    free_at = -np.inf

    for i in range(n):
        if in_pos:
            held = ts[i] - ts[entry_i]
            if held < min_hold:
                continue
            entry_cost = sell[entry_i]
            if entry_cost <= 0:
                in_pos = False
                continue
            spread_capture = (buy[entry_i] * (1 - tax) - sell[entry_i]) / entry_cost
            directional = (buy[i] - buy[entry_i]) * (1 - tax) / entry_cost
            total = spread_capture + directional

            hit_tp = directional >= take_profit
            hit_sl = directional <= -stop_loss
            timed_out = held >= max_hold
            if hit_tp or hit_sl or timed_out or i == n - 1:
                trades.append(
                    {
                        "entry_time": str(df["timestamp"].iloc[entry_i]),
                        "exit_time": str(df["timestamp"].iloc[i]),
                        "held_minutes": held / 60.0,
                        "entry_sell": sell[entry_i],
                        "entry_buy": buy[entry_i],
                        "exit_buy": buy[i],
                        "spread_capture": spread_capture,
                        "directional": directional,
                        "total": total,
                        "exit_reason": "tp" if hit_tp else "sl" if hit_sl else "timeout",
                    }
                )
                in_pos = False
                free_at = ts[i] + cooldown
            continue

        if signal[i] and sell[i] > 0 and ts[i] >= free_at:
            in_pos = True
            entry_i = i

    return pd.DataFrame(trades)


def _equity(returns):
    """Compounded equity, numerically safe and ruin-aware."""
    r = np.clip(np.asarray(returns, dtype=float), -0.9999, None)
    eq = np.cumprod(1.0 + r)
    ruined = bool(np.any(np.asarray(returns) <= -1.0))
    return eq, ruined


def _stats(r, prefix):
    r = np.asarray(r, dtype=float)
    wins = r > 0
    eq, ruined = _equity(r)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    total = float(eq[-1] - 1) if np.isfinite(eq[-1]) else float("inf")
    return {
        f"{prefix}_mean_per_trade_%": float(r.mean() * 100),
        f"{prefix}_median_per_trade_%": float(np.median(r) * 100),
        f"{prefix}_summed_%": float(r.sum() * 100),
        f"{prefix}_compounded_%": total * 100,
        f"{prefix}_compounding_ruined": ruined,
        f"{prefix}_win_rate_%": float(wins.mean() * 100),
        f"{prefix}_max_drawdown_%": float(dd.min() * 100),
    }


def summarize(trades, df, signal=None, tax=TAX, rng_seed=0):
    """Trade log -> the numbers that matter, with a random-entry control."""
    if trades is None or len(trades) == 0:
        return {"n_trades": 0, "note": "model never fired / no completed trades"}

    out = {"n_trades": int(len(trades))}
    out.update(_stats(trades["total"], "total"))
    out.update(_stats(trades["directional"], "directional"))
    out["spread_capture_mean_%"] = float(trades["spread_capture"].mean() * 100)
    out["avg_hold_minutes"] = float(trades["held_minutes"].mean())
    out["exit_reasons"] = {k: int(v) for k, v in trades["exit_reason"].value_counts().items()}

    span_min = (
        pd.to_datetime(df["timestamp"].iloc[-1]) - pd.to_datetime(df["timestamp"].iloc[0])
    ).total_seconds() / 60.0
    out["exposure_%"] = float(trades["held_minutes"].sum() / span_min * 100) if span_min else None
    out["trades_per_day"] = float(len(trades) / (span_min / 1440)) if span_min else None

    # Control: same number of entries, same rules, chosen at random. This is the
    # bar the model has to clear -- with a 12.8% spread, random entries also
    # "profit", so only the directional gap is evidence of skill.
    rng = np.random.default_rng(rng_seed)
    rand_sig = np.zeros(len(df), dtype=bool)
    k = min(int(signal.sum()) if signal is not None else len(trades), len(df))
    if k > 0:
        rand_sig[rng.choice(len(df), size=k, replace=False)] = True
    rt = simulate_trades(df, rand_sig)
    if len(rt):
        out["random_entry_control"] = {
            "n_trades": int(len(rt)),
            "total_mean_per_trade_%": float(rt["total"].mean() * 100),
            "directional_mean_per_trade_%": float(rt["directional"].mean() * 100),
        }
        out["directional_edge_vs_random_pp"] = float(
            (trades["directional"].mean() - rt["directional"].mean()) * 100
        )

    # buy and hold on the same side of the book, taxed once
    first, last = df["buy_price"].iloc[0], df["buy_price"].iloc[-1]
    out["buy_and_hold_%"] = float((last * (1 - tax) - first) / first * 100) if first else None

    # Fixed stake, no compounding. Compounding full capital across hundreds of
    # trades implies order-book depth that does not exist -- it produces
    # absurd numbers -- so this is the figure to quote. Coins are per one stake
    # unit re-used each trade, which is how a flipper actually operates.
    for stake in (1_000_000, 10_000_000):
        out[f"stake_{stake//1_000_000}M"] = {
            "total_profit_coins": float(trades["total"].sum() * stake),
            "model_attributable_coins": float(
                out.get("directional_edge_vs_random_pp", 0.0) / 100 * len(trades) * stake
            ),
            "spread_component_coins": float(trades["spread_capture"].sum() * stake),
            "worst_trade_coins": float(trades["total"].min() * stake),
        }
    return out


def record_metrics(item_id, payload, filename=None):
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = os.path.join(METRICS_DIR, filename or f"{item_id}_holdout.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    return path
