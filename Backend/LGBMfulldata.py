import os
import sys

# If the project root not in system's path
# Insert it in
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import optuna  # noqa: E402
import json  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import joblib  # noqa: E402
import lightgbm as lgb  # noqa: E402
import warnings  # noqa: E402
from Utils.event_utils import add_skyblock_time_features  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from Utils.data_utils import load_or_fetch_item_data  # noqa: E402
from Utils.mayor_utils import get_mayor_perks, match_mayor_perks  # noqa: E402
from Utils.load_proxies import load_proxies  # noqa: E402
from Utils.data_utils import configure_proxy_pool  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
from numba import njit, prange  # noqa: E402

warnings.filterwarnings("ignore")


# =========================================================
# Modelling configuration
# =========================================================
#
# Two labelling/objective schemes exist in this codebase:
#
#   LABEL_MODE = "median"  entry_label is the median forward return over the
#                          horizon, net of tax and spread-neutralised.
#   LABEL_MODE = "tpsl"    entry_label is the first-touch take-profit /
#                          stop-loss barrier outcome at +/-1.5% (PR #6).
#
#   OBJECTIVE = "regression"  predict the return itself, tuned with a
#                             sign-penalty-weighted RMSE.
#   OBJECTIVE = "binary"      predict P(entry_label >= SPIKE_THRESHOLD),
#                             tuned with log-loss (PR #6).
#
# Defaults are median+regression because that combination measured better on
# every item tested, on BOTH label definitions as ground truth -- including the
# tpsl one. Numbers and method: docs/PR6_VERIFICATION.md.
# Override per-run with BAZAAR_LABEL_MODE / BAZAAR_OBJECTIVE.

LABEL_MODE = os.environ.get("BAZAAR_LABEL_MODE", "median").strip().lower()
OBJECTIVE = os.environ.get("BAZAAR_OBJECTIVE", "regression").strip().lower()
SPIKE_THRESHOLD = 0.015  # "tradeable move" cutoff, used by OBJECTIVE="binary"
ENTRY_CONFIDENCE = 0.60  # min predicted probability to surface a signal (binary)


# =========================================================
# CPU budget
# =========================================================
#
# Optuna's n_jobs runs that many trials concurrently, and each trial's LightGBM
# independently defaults to *every* core. n_jobs=-1 therefore asks for
# cores x cores threads -- 64 on an 8-core laptop -- and the resulting
# oversubscription thrashes badly enough on a few hundred thousand rows to make
# the machine unusable (observed load average 29 on 8 cores).
#
# Budget the two together instead. Defaults to half the cores so a training run
# leaves the machine responsive; raise BAZAAR_CPU_BUDGET on a dedicated box.

_CPU_COUNT = os.cpu_count() or 4
CPU_BUDGET = max(1, int(os.environ.get("BAZAAR_CPU_BUDGET", max(1, _CPU_COUNT // 2))))
OPTUNA_JOBS = max(1, min(CPU_BUDGET, 4))
LGB_THREADS = max(1, CPU_BUDGET // OPTUNA_JOBS)

if LABEL_MODE not in ("median", "tpsl"):
    raise ValueError(f"LABEL_MODE must be 'median' or 'tpsl', got {LABEL_MODE!r}")
if OBJECTIVE not in ("regression", "binary"):
    raise ValueError(f"OBJECTIVE must be 'regression' or 'binary', got {OBJECTIVE!r}")


# =========================================================
# Data Cleaning (regression path)
# =========================================================


def clip_extreme_outliers(y, threshold=0.25):
    y = np.asarray(y)
    return np.clip(y, -threshold, threshold)


def remove_extremes(X, y, cutoff=0.5):
    mask = np.abs(y) <= cutoff
    return X[mask], y[mask]


# =========================================================
# Training device
# =========================================================

# LightGBM from pip is built without OpenCL, so "gpu" is fatal on a stock
# install. Probe once and fall back, overridable with BAZAAR_DEVICE=gpu|cpu.
_DEVICE_TYPE = None


def get_device_type():
    """Return a LightGBM device_type this machine can actually train on."""
    global _DEVICE_TYPE
    if _DEVICE_TYPE is not None:
        return _DEVICE_TYPE

    forced = os.environ.get("BAZAAR_DEVICE", "").strip().lower()
    if forced in ("cpu", "gpu", "cuda"):
        _DEVICE_TYPE = forced
        return _DEVICE_TYPE

    # LightGBM logs "[Fatal] GPU Tree Learner was not enabled" from C++ straight
    # to fd 1, past verbosity and past Python's stdout, so mute the fd itself.
    saved_out, saved_err = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        probe = np.random.default_rng(0).random((64, 4))
        lgb.train(
            {"objective": "binary", "device_type": "gpu", "verbosity": -1,
             "min_data_in_leaf": 1, "max_bin": 63},
            lgb.Dataset(probe, label=(probe[:, 0] > 0.5).astype(int)),
            num_boost_round=1,
        )
        _DEVICE_TYPE = "gpu"
    except Exception:
        _DEVICE_TYPE = "cpu"
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)
    print(f"  → LightGBM device_type: {_DEVICE_TYPE}")
    return _DEVICE_TYPE


# =========================================================
# Data Dump
# =========================================================


def clean_dump(obj, path):
    """Atomic write with fsync for durability"""
    tmp = path + ".tmp"
    joblib.dump(obj, tmp)

    try:
        with open(tmp, "r+b") as f:
            f.flush()
            os.fsync(f.fileno())
    except (OSError, IOError):
        pass

    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp, path)


# =========================================================
# Data Cleaning
# =========================================================
def clean_infinite_values(X):
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e8, neginf=-1e8)
    return np.clip(X, -1e8, 1e8)


# =========================================================
# Feature Engineering
# =========================================================


def add_time_features(df, ts_col="timestamp"):
    dt = pd.to_datetime(df[ts_col])
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["dayofweek"] = dt.dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df["delta_minutes"] = df["timestamp"].diff().dt.total_seconds().fillna(0) / 60
    return df


def build_lagged_features(
    df, price_col="buy_price", vol_col="buy_volume", lags=(1, 2, 3, 6, 12), prefix=""
):
    ret = df[price_col].pct_change()
    df[f"{prefix}ret"] = ret

    for lag in lags:
        df[f"{prefix}ret_lag_{lag}"] = ret.shift(lag)
        df[f"{prefix}price_lag_{lag}"] = df[price_col].shift(lag)
        df[f"{prefix}vol_lag_{lag}"] = df[vol_col].shift(lag)

    df[f"{prefix}roll_mean_6"] = ret.rolling(6).mean()
    df[f"{prefix}roll_std_6"] = ret.rolling(6).std()
    df[f"{prefix}mom_6"] = ret.rolling(6).sum()

    return df


def prepare_dataframe_from_raw(data, mayor_data=None):
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    required_cols = [
        "timestamp",
        "buy",
        "sell",
        "buyVolume",
        "sellVolume",
        "maxBuy",
        "minBuy",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0

    # The feed mixes fractional-second precision (".038", ".03", ".0" and none
    # at all). A bare pd.to_datetime infers one format from the first element
    # and silently coerces every row that doesn't match to NaT, which dropna
    # then discards -- 99 rows per 100k here, each one punching a hole in the
    # forward window of every row within the horizon behind it. ISO8601 parses
    # the mixed precision instead of guessing.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce", format="ISO8601"
    )
    dropped = int(df["timestamp"].isna().sum())
    if dropped:
        print(f"  ⚠ {dropped} rows dropped: unparseable timestamps")
    df = df.dropna(subset=["timestamp"])

    cols_to_float = {
        "buy": "buy_price",
        "sell": "sell_price",
        "buyVolume": "buy_volume",
        "sellVolume": "sell_volume",
        "maxBuy": "max_buy",
        "minBuy": "min_buy",
    }

    for old_col, new_col in cols_to_float.items():
        df[new_col] = (
            pd.to_numeric(df[old_col], errors="coerce").fillna(0.0).astype(float)
        )

    df = df.drop(columns=list(cols_to_float.keys()))

    if df.empty:
        return df

    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(timezone.utc)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(timezone.utc)

    df = df.sort_values("timestamp").reset_index(drop=True)

    if mayor_data is not None and len(mayor_data) > 0:
        mayor_df = pd.DataFrame(mayor_data)
        if "start_date" in mayor_df.columns and "perks" in mayor_df.columns:
            perks_df = pd.DataFrame(mayor_df["perks"].tolist(), index=mayor_df.index)
            perks_df.columns = [f"mayor_{i}" for i in range(perks_df.shape[1])]

            mayor_df = pd.concat([mayor_df.drop(columns=["perks"]), perks_df], axis=1)
            mayor_df["start_date"] = pd.to_datetime(mayor_df["start_date"], utc=True)
            mayor_df = mayor_df.sort_values("start_date")

            df = pd.merge_asof(
                df,
                mayor_df,
                left_on="timestamp",
                right_on="start_date",
                direction="backward",
            )
            df = df.drop(columns=["start_date"])

            mayor_cols = [col for col in df.columns if col.startswith("mayor_")]
            df[mayor_cols] = df[mayor_cols].fillna(0.0)

    df = add_time_features(df)
    df = build_lagged_features(
        df, price_col="buy_price", vol_col="buy_volume", prefix="buy_"
    )
    df = add_skyblock_time_features(df, ts_col="timestamp")
    return df


# parallel=True is deliberately off: the body carries `ret` out of the inner
# loop into the `not hit_target` fallback, which numba's parfor reduction pass
# cannot analyse -- it recurses until RecursionError and the module fails to
# import. Serial njit still gives ~2.7x over the pure-Python loop.
@njit(fastmath=True, cache=True, parallel=False)
def _compute_targets_jit(
    df_len,
    ts,
    buy_prices,
    sell_prices,
    initial_gaps,
    horizon_sec,
    tax,
):
    n = df_len
    expected_return = np.zeros(n)
    median_return = np.zeros(n)
    profit_prob = np.zeros(n)
    time_to_first_up = np.zeros(n)
    time_to_first_down = np.zeros(n)
    time_to_max = np.zeros(n)
    time_to_min = np.zeros(n)
    max_profit_list = np.zeros(n)
    max_loss_list = np.zeros(n)
    risk_reward_list = np.zeros(n)
    win_rate_1pct_list = np.zeros(n)
    win_rate_2pct_list = np.zeros(n)
    mae_list = np.zeros(n)
    mfe_list = np.zeros(n)
    profitable_1pct_list = np.zeros(n, dtype=np.int64)
    profitable_2pct_list = np.zeros(n, dtype=np.int64)

    min_delay = 10 * 60

    for i in prange(n):
        entry_price = sell_prices[i]
        initial_gap = initial_gaps[i]
        start_ts = ts[i]

        max_profit = -np.inf
        max_loss = np.inf
        t_max_rel_idx = -1
        t_min_rel_idx = -1

        first_up_idx = -1
        first_down_idx = -1

        count = 0
        pos_count = 0
        win_1pct_count = 0
        win_2pct_count = 0

        for j in range(i, n):
            time_delta = ts[j] - start_ts
            if time_delta > horizon_sec:
                break

            if time_delta > min_delay:
                ret = (buy_prices[j] * (1 - tax) - entry_price - initial_gap) / (
                    entry_price + 1e-9
                )
                count += 1

                if ret > 0:
                    pos_count += 1
                    if first_up_idx == -1:
                        first_up_idx = j
                if ret < 0:
                    if first_down_idx == -1:
                        first_down_idx = j

                if ret > max_profit:
                    max_profit = ret
                    t_max_rel_idx = j

                if ret < max_loss:
                    max_loss = ret
                    t_min_rel_idx = j

                if ret >= 0.01:
                    win_1pct_count += 1
                if ret >= 0.02:
                    win_2pct_count += 1

        if count == 0:
            continue

        mae_running = np.inf
        # Chronological TP / SL check
        # We define a 1.5% take profit and 1.5% stop loss threshold
        tp_threshold = 0.015
        sl_threshold = -0.015

        final_return = 0.0
        hit_target = False

        # Both labels are computed in one pass so LABEL_MODE can switch between
        # them without recomputing targets. The TP/SL scan no longer breaks on
        # first touch -- it freezes final_return/mae behind `hit_target` instead
        # -- so the window can be collected in full for the median.
        window = np.empty(count)
        k = 0

        for j in range(i, n):
            time_delta = ts[j] - start_ts
            if time_delta > horizon_sec:
                break

            if time_delta > min_delay:
                ret = (buy_prices[j] * (1 - tax) - entry_price - initial_gap) / (
                    entry_price + 1e-9
                )

                if k < count:
                    window[k] = ret
                    k += 1

                if not hit_target:
                    if ret < mae_running:
                        mae_running = ret

                    if ret >= tp_threshold:
                        final_return = ret
                        hit_target = True
                    elif ret <= sl_threshold:
                        final_return = ret
                        hit_target = True

        if not hit_target and count > 0:
            # If we never hit TP or SL, our expected return is simply the last return we saw
            final_return = ret

        if k > 0:
            srt = np.sort(window[:k])
            if k % 2 == 1:
                median_return[i] = srt[k // 2]
            else:
                median_return[i] = 0.5 * (srt[k // 2 - 1] + srt[k // 2])

        mae_val = mae_running if mae_running != np.inf else 0.0
        expected_return[i] = final_return

        profit_prob[i] = pos_count / count

        if first_up_idx != -1:
            time_to_first_up[i] = ts[first_up_idx] - start_ts
        if first_down_idx != -1:
            time_to_first_down[i] = ts[first_down_idx] - start_ts

        if t_max_rel_idx != -1:
            time_to_max[i] = ts[t_max_rel_idx] - start_ts
            time_to_min[i] = ts[t_min_rel_idx] - start_ts

        max_profit_list[i] = max_profit
        max_loss_list[i] = max_loss
        risk_reward_list[i] = max_profit / abs(max_loss) if max_loss < 0 else max_profit

        win_rate_1pct_list[i] = win_1pct_count / count
        win_rate_2pct_list[i] = win_2pct_count / count

        mae_list[i] = mae_val
        mfe_list[i] = max_profit

        profitable_1pct_list[i] = int(max_profit >= 0.01)
        profitable_2pct_list[i] = int(max_profit >= 0.02)

    return (
        expected_return,
        median_return,
        profit_prob,
        time_to_first_up,
        time_to_first_down,
        time_to_max,
        time_to_min,
        max_profit_list,
        max_loss_list,
        risk_reward_list,
        win_rate_1pct_list,
        win_rate_2pct_list,
        mae_list,
        mfe_list,
        profitable_1pct_list,
        profitable_2pct_list,
    )


def build_entry_targets(df, horizon_minutes=180, tax=0.0125):
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"]).astype("int64") // 10**9
    ts = ts.values
    horizon_sec = horizon_minutes * 60

    buy_prices = df["buy_price"].values
    sell_prices = df["sell_price"].values

    initial_gaps = buy_prices * (1 - tax) - sell_prices

    (
        expected_return,
        median_return,
        profit_prob,
        time_to_first_up,
        time_to_first_down,
        time_to_max,
        time_to_min,
        max_profit_list,
        max_loss_list,
        risk_reward_list,
        win_rate_1pct_list,
        win_rate_2pct_list,
        mae_list,
        mfe_list,
        profitable_1pct_list,
        profitable_2pct_list,
    ) = _compute_targets_jit(
        len(df),
        ts,
        buy_prices,
        sell_prices,
        initial_gaps,
        horizon_sec,
        tax,
    )

    returns_last_5min = df["buy_price"].pct_change(periods=5)
    returns_last_15min = df["buy_price"].pct_change(periods=15)
    price_vs_5min_high = df["buy_price"] / df["buy_price"].rolling(5).max()
    price_vs_5min_low = df["buy_price"] / df["buy_price"].rolling(5).min()
    price_volatility = (
        df["buy_price"].rolling(20).std() / df["buy_price"].rolling(20).mean()
    )
    spread_volatility = (df["buy_price"] - df["sell_price"]).rolling(20).std()
    spread_pct = (df["buy_price"] - df["sell_price"]) / df["sell_price"]
    spread_momentum = spread_pct.diff()

    df["returns_last_5min"] = returns_last_5min
    df["returns_last_15min"] = returns_last_15min
    df["price_vs_5min_high"] = price_vs_5min_high
    df["price_vs_5min_low"] = price_vs_5min_low
    df["price_volatility"] = price_volatility
    df["spread_volatility"] = spread_volatility
    df["spread_pct"] = spread_pct
    df["spread_momentum"] = spread_momentum
    df["max_profit"] = max_profit_list
    df["max_loss"] = max_loss_list
    df["risk_reward"] = risk_reward_list
    df["win_rate_1pct"] = win_rate_1pct_list
    df["win_rate_2pct"] = win_rate_2pct_list
    df["mae"] = mae_list
    df["mfe"] = mfe_list
    df["profitable_1pct"] = profitable_1pct_list
    df["profitable_2pct"] = profitable_2pct_list
    # Both label definitions are kept on the frame so a CSV generated under one
    # LABEL_MODE stays usable under the other; entry_label is whichever mode is
    # active. See docs/PR6_VERIFICATION.md for the A/B behind the default.
    df["entry_label_tpsl"] = expected_return
    df["entry_label_median"] = median_return
    df["entry_label"] = median_return if LABEL_MODE == "median" else expected_return
    df["profit_prob"] = profit_prob
    df["time_to_first_up"] = time_to_first_up
    df["time_to_first_down"] = time_to_first_down
    df["time_to_max"] = time_to_max
    df["time_to_min"] = time_to_min

    # Remove extreme outliers (>200% return)
    df = df[np.abs(df["entry_label"]) <= 2.0]

    return df


def load_entry_targets(item_id):
    csv_directory = os.path.join(project_root, "csv files")
    df = pd.read_csv(os.path.join(csv_directory, f"{item_id}_debug_data.csv"))

    # parse_dates= silently gives up on this column and leaves it as strings:
    # the CSV round-trip preserves the feed's mixed fractional-second precision,
    # and pandas infers one format from the first row. Failing here is quiet --
    # the trainer drops 'timestamp' from the features anyway, so it only
    # surfaces later wherever a .dt accessor is used.
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format="ISO8601", utc=True, errors="coerce"
        )

    # A cached CSV froze entry_label at whichever LABEL_MODE generated it, so
    # without this the flag would silently do nothing whenever a CSV already
    # exists. Both variants are stored, so re-point entry_label at the active
    # one instead of forcing a regenerate.
    wanted = f"entry_label_{LABEL_MODE}"
    if wanted in df.columns:
        df["entry_label"] = df[wanted]
    elif "entry_label" in df.columns:
        print(
            f"  ⚠ {item_id}: cached CSV predates LABEL_MODE and has no "
            f"'{wanted}' column; using its stored entry_label as-is. "
            f"Delete 'csv files/{item_id}_debug_data.csv' to regenerate."
        )
    return df


# =========================================================
# Optuna Objective
# =========================================================


def _entry_objective_regression(trial, X, y):
    """Sign-penalty-weighted RMSE: a wrong-direction prediction costs 1-5x a
    right-direction one of the same magnitude, because direction is what the
    trade decision actually turns on.

    The scaler is fit on the training split only -- fitting it on all of X
    before splitting leaked validation distribution into tuning.
    """
    split_idx = int(len(X) * 0.8)
    X_train_raw, X_val_raw = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    if len(X_train_raw) == 0 or len(X_val_raw) == 0:
        return 9999.0

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    clip_thr = trial.suggest_float("label_clip", 0.01, 0.5, log=True)
    y_train = np.clip(y_train, -clip_thr, clip_thr)
    y_val = np.clip(y_val, -clip_thr, clip_thr)

    sign_penalty = trial.suggest_float("sign_penalty", 1.0, 5.0)

    params = {
        "objective": "regression",
        "device_type": "cpu",
        "metric": "rmse",
        "learning_rate": trial.suggest_float("lr", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 3,
        "verbosity": -1,
        "num_threads": LGB_THREADS,
    }

    try:
        model = lgb.train(
            params, lgb.Dataset(X_train, label=y_train), num_boost_round=300
        )
    except Exception as e:
        print(f"  ⚠ Unexpected error during trial: {e}")
        return 9999.0

    preds = model.predict(X_val)
    sq_errors = (preds - y_val) ** 2
    weights = np.ones_like(sq_errors)
    weights[np.sign(preds) != np.sign(y_val)] = sign_penalty
    return float(np.sqrt(np.mean(weights * sq_errors)))


def _entry_objective_binary(trial, X, y):
    split_idx = int(len(X) * 0.8)

    X_train_raw, X_val_raw = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    if len(X_train_raw) == 0 or len(X_val_raw) == 0 or len(np.unique(y_train)) < 2:
        return 9999.0

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    params = {
        "objective": "binary",
        "device_type": "cpu",
        "metric": "binary_logloss",
        "scale_pos_weight": scale_pos_weight,
        "learning_rate": trial.suggest_float("lr", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 3,
        "verbosity": -1,
        "num_threads": LGB_THREADS,
    }

    def pruning_callback(env):
        if env.iteration < 50:
            return
        preds = env.model.predict(X_val)
        eps = 1e-15
        preds = np.clip(preds, eps, 1 - eps)
        loss = -np.mean(y_val * np.log(preds) + (1 - y_val) * np.log(1 - preds))
        trial.report(loss, step=env.iteration)
        if trial.should_prune():
            raise optuna.TrialPruned()

    dtrain = lgb.Dataset(X_train, label=y_train)

    try:
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=100,
            callbacks=[pruning_callback],
        )
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"  ⚠ Unexpected error during trial: {e}")
        return 9999.0

    preds = model.predict(X_val)
    eps = 1e-15
    preds = np.clip(preds, eps, 1 - eps)
    loss = -np.mean(y_val * np.log(preds) + (1 - y_val) * np.log(1 - preds))

    return loss


def entry_objective(trial, X, y):
    """Dispatch to the objective selected by OBJECTIVE."""
    if OBJECTIVE == "binary":
        return _entry_objective_binary(trial, X, y)
    return _entry_objective_regression(trial, X, y)


# =========================================================
# GENERATE CSV FILES
# =========================================================


def generate_csv_files(
    item_id,
    enable_fetch_if_missing,
    enable_update_with_new_data,
    mayor_data=None,
):
    if mayor_data is None:
        mayor_data = get_mayor_perks()

    data = load_or_fetch_item_data(
        item_id,
        enable_fetch_if_missing,
        enable_update_with_new_data,
    )
    df = prepare_dataframe_from_raw(data, mayor_data)
    df = add_skyblock_time_features(df)
    df = build_lagged_features(df)
    df = add_time_features(df)
    df = build_entry_targets(df)

    csv_directory = os.path.join(project_root, "csv files")
    os.makedirs(csv_directory, exist_ok=True)

    csv_path = os.path.join(csv_directory, f"{item_id}_debug_data.csv")
    df.to_csv(csv_path, index=False)

    return df


# =========================================================
# Training
# =========================================================


def train_model_system(
    item_id, fetch_if_missing, update_with_new_data, mayor_data=None
):
    if os.path.exists(
        os.path.join(project_root, "csv files", f"{item_id}_debug_data.csv")
    ):
        print(f"✓ CSV file for {item_id} already exists")
        df = load_entry_targets(item_id)
    else:
        print(f"✗ CSV file for {item_id} does not exist")
        df = generate_csv_files(
            item_id,
            fetch_if_missing,
            update_with_new_data,
            mayor_data=mayor_data,
        )

    future_cols = {
        "entry_label",
        # both label variants live on the frame now -- leaving either in the
        # feature set would hand the model the answer directly.
        "entry_label_tpsl",
        "entry_label_median",
        "max_profit",
        "max_loss",
        "risk_reward",
        "win_rate_1pct",
        "win_rate_2pct",
        "mae",
        "mfe",
        "profitable_1pct",
        "profitable_2pct",
        "profit_prob",
        "time_to_first_up",
        "time_to_first_down",
        "time_to_max",
        "time_to_min",
    }
    exclude = {"timestamp"} | future_cols

    feature_cols = [c for c in df.columns if c not in exclude]

    X = clean_infinite_values(df[feature_cols].values)
    y_raw = df["entry_label"].values

    print(f"  → LABEL_MODE={LABEL_MODE}  OBJECTIVE={OBJECTIVE}")

    if OBJECTIVE == "binary":
        # Convert problem to Yes/No: "Will this go up by at least 1.5%?"
        y = (y_raw >= SPIKE_THRESHOLD).astype(int)

        split_idx = int(len(X) * 0.8)
        y_train_split = y[:split_idx]
        y_val_split = y[split_idx:]

        if np.sum(y_train_split) < 20 or np.sum(y_val_split) < 5:
            print(
                f"  ⚠ Skipping {item_id}: Not enough positive examples in train/val sets (Train: {np.sum(y_train_split)}, Val: {np.sum(y_val_split)})."
            )
            return
    else:
        X, y = remove_extremes(X, y_raw, cutoff=0.5)
        if len(y) == 0:
            print(f"  ⚠ Skipping {item_id}: no rows left after extreme removal.")
            return

    # Trial runs
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=50,
        ),
    )
    study.optimize(
        lambda t: entry_objective(t, X, y),
        n_trials=30,
        n_jobs=OPTUNA_JOBS,  # bounded jointly with LGB_THREADS, see CPU budget
    )

    params = study.best_params
    if not params:
        print(f"  ⚠ Skipping {item_id}: Optuna could not find any valid parameters.")
        return

    if OBJECTIVE == "binary":
        # Handle class imbalance
        neg_count = np.sum(y == 0)
        pos_count = np.sum(y == 1)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        params.update(
            {
                "objective": "binary",
                "device_type": get_device_type(),
                "metric": "binary_logloss",
                "scale_pos_weight": scale_pos_weight,
                "min_data_in_leaf": 50,  # Prevents GPU decision tree split crash
                "max_bin": 63,  # Optimized for stable GPU training
                "verbosity": -1,
                "num_threads": LGB_THREADS,
            }
        )
    else:
        # search-time only, not LightGBM parameters
        best_clip = params.pop("label_clip", 0.25)
        params.pop("sign_penalty", None)
        y = clip_extreme_outliers(y, threshold=best_clip)
        params.update(
            {
                "objective": "regression",
                "device_type": get_device_type(),
                "metric": "rmse",
                "min_data_in_leaf": 50,
                "max_bin": 63,
                "verbosity": -1,
                "num_threads": LGB_THREADS,
            }
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Final train
    model = lgb.train(
        params,
        lgb.Dataset(X_scaled, label=y),
        num_boost_round=400,
    )
    model_dir = os.path.join(project_root, "Model_Files")
    os.makedirs(model_dir, exist_ok=True)

    base = os.path.join(model_dir, str(item_id))

    try:
        clean_dump(model, base + "_entry_model.pkl")
        clean_dump(scaler, base + "_entry_scaler.pkl")
        clean_dump(feature_cols, base + "_entry_features.pkl")
        print(f"✓ Successfully saved models for {item_id}")
    except Exception as e:
        print(f"✗ Error saving files for {item_id}: {e}")


# =========================================================
# Test Train Setup for Model Accuracy Metrics
# =========================================================


def test_train_model_system(
    item_id, fetch_if_missing, update_with_new_data, mayor_data=None
):
    if os.path.exists(
        os.path.join(project_root, "csv files", f"{item_id}_debug_data.csv")
    ):
        print(f"✓ CSV file for {item_id} already exists")
        df = load_entry_targets(item_id)
    else:
        print(f"✗ CSV file for {item_id} does not exist")
        df = generate_csv_files(
            item_id,
            fetch_if_missing,
            update_with_new_data,
            mayor_data=mayor_data,
        )

    tested_metrics_dict = {}

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    future_cols = {
        "entry_label",
        # both label variants live on the frame now -- leaving either in the
        # feature set would hand the model the answer directly.
        "entry_label_tpsl",
        "entry_label_median",
        "max_profit",
        "max_loss",
        "risk_reward",
        "win_rate_1pct",
        "win_rate_2pct",
        "mae",
        "mfe",
        "profitable_1pct",
        "profitable_2pct",
        "profit_prob",
        "time_to_first_up",
        "time_to_first_down",
        "time_to_max",
        "time_to_min",
    }

    exclude = {"timestamp"} | future_cols
    feature_cols = [c for c in df.columns if c not in exclude]

    X_train = clean_infinite_values(train_df[feature_cols].values)
    y_train_raw = train_df["entry_label"].values
    X_val = clean_infinite_values(val_df[feature_cols].values)
    y_val_raw = val_df["entry_label"].values

    print(f"  → LABEL_MODE={LABEL_MODE}  OBJECTIVE={OBJECTIVE}")

    if OBJECTIVE == "binary":
        y_train = (y_train_raw >= SPIKE_THRESHOLD).astype(int)
        y_val = (y_val_raw >= SPIKE_THRESHOLD).astype(int)

        if np.sum(y_train) < 20 or np.sum(y_val) < 5:
            print(
                f"  ⚠ Skipping {item_id}: Not enough positive examples in train/val sets (Train: {np.sum(y_train)}, Val: {np.sum(y_val)})."
            )
            return
    else:
        # Drop unmodellable tails before scaling, so the rows fed to Optuna and
        # the rows fed to the final fit stay aligned.
        X_train, y_train = remove_extremes(X_train, y_train_raw, cutoff=0.5)
        X_val, y_val = remove_extremes(X_val, y_val_raw, cutoff=0.5)
        if len(y_train) == 0 or len(y_val) == 0:
            print(f"  ⚠ Skipping {item_id}: no rows left after extreme removal.")
            return

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    if OBJECTIVE == "binary":
        print(f"Train spikes: {np.sum(y_train)} / {len(y_train)}")
        print(f"Val spikes: {np.sum(y_val)} / {len(y_val)}")
    else:
        print(
            f"y_train: {np.mean(y_train > 0) * 100:.1f}% positive, "
            f"median={np.median(y_train):+.5f}, n={len(y_train):,}"
        )
        print(
            f"y_val:   {np.mean(y_val > 0) * 100:.1f}% positive, "
            f"median={np.median(y_val):+.5f}, n={len(y_val):,}"
        )

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=50,
        ),
    )
    study.optimize(
        lambda t: entry_objective(t, X_train, y_train),
        n_trials=30,
        n_jobs=OPTUNA_JOBS,  # bounded jointly with LGB_THREADS, see CPU budget
    )

    params = study.best_params
    if not params:
        print(f"  ⚠ Skipping {item_id}: Optuna could not find any valid parameters.")
        return

    if OBJECTIVE == "binary":
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        params.update(
            {
                "objective": "binary",
                "device_type": get_device_type(),
                "metric": "binary_logloss",
                "scale_pos_weight": scale_pos_weight,
                "min_data_in_leaf": 50,  # Prevent GPU decision tree split crash
                "max_bin": 63,  # Stable and fast on GPU
                "verbosity": -1,
                "num_threads": LGB_THREADS,
            }
        )
    else:
        # label_clip / sign_penalty are search-time only -- they shape the tuning
        # signal, they are not LightGBM parameters.
        best_clip = params.pop("label_clip", 0.25)
        params.pop("sign_penalty", None)
        y_train = clip_extreme_outliers(y_train, threshold=best_clip)
        y_val = clip_extreme_outliers(y_val, threshold=best_clip)
        params.update(
            {
                "objective": "regression",
                "device_type": get_device_type(),
                "metric": "rmse",
                "min_data_in_leaf": 50,
                "max_bin": 63,
                "verbosity": -1,
                "num_threads": LGB_THREADS,
            }
        )

    model = lgb.train(
        params,
        lgb.Dataset(X_train_scaled, label=y_train),
        num_boost_round=400,
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)
    print("\nTop 20 Features by Gain:")
    print(importance_df.head(20))

    y_score = model.predict(X_val_scaled)

    # Ground truth for the "was this a tradeable move" question is the same in
    # both modes, so ROC AUC is directly comparable across OBJECTIVE settings.
    y_true_spike = (
        y_val.astype(int)
        if OBJECTIVE == "binary"
        else (y_val >= SPIKE_THRESHOLD).astype(int)
    )
    fire = y_score >= 0.5 if OBJECTIVE == "binary" else y_score > 0

    try:
        acc = accuracy_score(y_true_spike, fire.astype(int))
        prec = precision_score(y_true_spike, fire.astype(int), zero_division=0)
        rec = recall_score(y_true_spike, fire.astype(int), zero_division=0)
        f1 = f1_score(y_true_spike, fire.astype(int), zero_division=0)
        auc = roc_auc_score(y_true_spike, y_score)
    except Exception:
        acc, prec, rec, f1, auc = 0, 0, 0, 0, 0

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f} (When the bot buys, how often is it a real spike?)")
    print(f"Recall:    {rec:.4f} (Out of all real spikes, how many did the bot catch?)")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")

    tested_metrics_dict["label_mode"] = LABEL_MODE
    tested_metrics_dict["objective"] = OBJECTIVE
    tested_metrics_dict["accuracy"] = acc
    tested_metrics_dict["precision"] = prec
    tested_metrics_dict["recall"] = rec
    tested_metrics_dict["f1"] = f1
    tested_metrics_dict["roc_auc"] = auc
    tested_metrics_dict["total_validation_samples"] = int(len(y_val))
    tested_metrics_dict["actual_spikes_in_validation"] = int(np.sum(y_true_spike))

    if OBJECTIVE == "regression":
        rmse = float(np.sqrt(np.mean((y_score - y_val) ** 2)))
        mae_err = float(np.mean(np.abs(y_score - y_val)))
        sign_acc = float(np.mean(np.sign(y_score) == np.sign(y_val)))
        mask = y_val > 0.01
        safe_sign = (
            float(np.mean((y_score[mask] > 0) == (y_val[mask] > 0)))
            if mask.any()
            else 0.0
        )
        print(f"RMSE:      {rmse:.6f}")
        print(f"MAE:       {mae_err:.6f}")
        print(f"Sign acc:  {sign_acc:.4f}")
        print(f"Safe sign: {safe_sign:.4f} (restricted to true moves > 1%)")
        tested_metrics_dict["rmse"] = rmse
        tested_metrics_dict["mae"] = mae_err
        tested_metrics_dict["sign_accuracy"] = sign_acc
        tested_metrics_dict["safe_sign_accuracy"] = safe_sign

    with open(
        os.path.join(project_root, "Model_Files", f"{item_id}_test_train_metrics.json"),
        "w",
    ) as f:
        json.dump(tested_metrics_dict, f, indent=4)


# =========================================================
# FUTURE PREDICTIONS (MULTI-TIMESTAMP)
# =========================================================


def predict_entries(
    model,
    scaler,
    feature_cols,
    item_id,
    mayor_data=None,
    horizon_hours=3,
    step_minutes=5,
):
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=horizon_hours)
    start_str = start.strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")
    end_str = now.strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")
    base_url = "https://sky.coflnet.com/api/bazaar"
    url = f"{base_url}/{item_id}/history?start={start_str}&end={end_str}"

    from Utils.data_utils import _get_session, _check_rate_limit, _get_next_proxy
    import time

    session = _get_session()
    raw = None
    max_retries = 3
    for attempt in range(max_retries):
        _check_rate_limit()
        proxy_str = _get_next_proxy()
        proxies = {"http": proxy_str, "https": proxy_str} if proxy_str else None
        try:
            resp = session.get(url, proxies=proxies, timeout=10)
            resp.raise_for_status()
            raw = resp.json()
            break
        except Exception as e:
            print(f"      → predict_entries fetch failed for {item_id}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return []

    if not raw:
        return []

    df = prepare_dataframe_from_raw(raw, mayor_data)
    if df.empty:
        return []

    last_row = df.iloc[-1:].copy()

    future_times = pd.date_range(
        start=now,
        periods=int(horizon_hours * 60 / step_minutes),
        freq=f"{step_minutes}min",
    )

    preds = []
    scores = []
    for ts in future_times:
        row = last_row.copy()
        row["timestamp"] = ts

        if mayor_data is not None:
            perks = match_mayor_perks(ts, mayor_data)
            for i, v in enumerate(perks):
                row[f"mayor_{i}"] = v

        for c in feature_cols:
            if c not in row.columns:
                row[c] = 0.0
        row[feature_cols] = row[feature_cols].fillna(0.0)

        X = clean_infinite_values(row[feature_cols].values)
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)[0]
        row["entry_score"] = y_pred
        row["timestamp"] = ts.isoformat()
        scores.append(y_pred)
        preds.append(
            row[["timestamp", "buy_price", "sell_price", "entry_score"]].to_dict(
                orient="records"
            )[0]
        )

    return preds


# =========================================================
# ANALYZE TOP PREDICTIONS
# =========================================================


def analyze_entries(pred_list, top_n=None):
    """Rank entry signals by how soon they fire, then by confidence.

    ``top_n`` truncates the result. /investments passes the cached
    ``{item_id, timestamp, entries}`` wrappers rather than bare entries, so
    those are flattened here -- previously that endpoint raised TypeError on
    the missing kwarg, and would still have returned [] once it was added
    because ``entry_score`` lives on the nested entries, not the wrapper.
    """
    if not pred_list:
        return []

    flattened = []
    for e in pred_list:
        if isinstance(e, dict) and "entries" in e and "entry_score" not in e:
            for inner in e.get("entries") or []:
                merged = dict(inner)
                merged.setdefault("item_id", e.get("item_id"))
                flattened.append(merged)
        else:
            flattened.append(e)
    pred_list = flattened

    now = datetime.now(timezone.utc)
    enriched = []

    for e in pred_list:
        try:
            score = float(e.get("entry_score", 0.0))
        except Exception:
            continue

        # Under OBJECTIVE="binary" the score is a probability, so require 60%
        # confidence. Under "regression" it is a predicted return, where the
        # entry rule is simply "expected to go up" -- applying the 0.60 cutoff
        # there would reject every signal, since returns sit around 0.01-0.05.
        if OBJECTIVE == "binary":
            if score < ENTRY_CONFIDENCE:
                continue
        elif score <= 0:
            continue

        ts_str = e.get("timestamp")
        if not ts_str:
            continue

        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception:
            continue

        delta_minutes = (ts - now).total_seconds() / 60.0
        if delta_minutes < 0:
            continue

        enriched_entry = dict(e)
        enriched_entry["delta_minutes"] = float(delta_minutes)
        enriched.append(enriched_entry)

    enriched.sort(key=lambda x: (x["delta_minutes"], -x["entry_score"]))

    return enriched[:top_n] if top_n else enriched


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    fetch_if_missing = True
    update_with_new_data = True

    proxies = load_proxies("proxies.txt")
    configure_proxy_pool(proxies)

    csv_directory = os.path.join(project_root, "csv files")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(
        script_dir,
        "bazaar_full_items_ids.json",
    )
    with open(file_path) as f:
        items = json.load(f)

    print("Fetching mayor data...")
    global_mayor_data = get_mayor_perks()

    for entry in items:
        # We switch to test_train_model_system here so you can verify the new accuracy metrics
        test_train_model_system(
            entry,
            fetch_if_missing,
            update_with_new_data,
            mayor_data=global_mayor_data,
        )
