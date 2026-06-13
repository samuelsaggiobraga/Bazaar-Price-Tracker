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

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
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


@njit(nopython=True, fastmath=True, cache=True, parallel=True)
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

        for j in range(i, n):
            time_delta = ts[j] - start_ts
            if time_delta > horizon_sec:
                break

            if time_delta > min_delay:
                ret = (buy_prices[j] * (1 - tax) - entry_price - initial_gap) / (
                    entry_price + 1e-9
                )

                if ret < mae_running:
                    mae_running = ret

                if ret >= tp_threshold:
                    final_return = ret
                    hit_target = True
                    break
                elif ret <= sl_threshold:
                    final_return = ret
                    hit_target = True
                    break

        if not hit_target and count > 0:
            # If we never hit TP or SL, our expected return is simply the last return we saw
            final_return = ret

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
    df["entry_label"] = expected_return
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
    df = pd.read_csv(
        os.path.join(csv_directory, f"{item_id}_debug_data.csv"),
        parse_dates=["timestamp"],
    )
    return df


# =========================================================
# Optuna Objective (Entry Classification)
# =========================================================


def entry_objective(trial, X, y):
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

    # === CLASSIFICATION TARGET ===
    # Convert problem to Yes/No: "Will this go up by at least 1.5%?"
    y = (y_raw >= 0.015).astype(int)
    # =============================

    split_idx = int(len(X) * 0.8)
    y_train_split = y[:split_idx]
    y_val_split = y[split_idx:]

    if np.sum(y_train_split) < 20 or np.sum(y_val_split) < 5:
        print(
            f"  ⚠ Skipping {item_id}: Not enough positive examples in train/val sets (Train: {np.sum(y_train_split)}, Val: {np.sum(y_val_split)})."
        )
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
        n_jobs=-1,  # Parralelize trials across all available CPU cores
    )

    params = study.best_params
    if not params:
        print(f"  ⚠ Skipping {item_id}: Optuna could not find any valid parameters.")
        return

    # Handle class imbalance
    neg_count = np.sum(y == 0)
    pos_count = np.sum(y == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    params.update(
        {
            "objective": "binary",
            "device_type": "gpu",
            "metric": "binary_logloss",
            "scale_pos_weight": scale_pos_weight,
            "min_data_in_leaf": 50,  # Prevents GPU decision tree split crash
            "max_bin": 63,  # Optimized for stable GPU training
            "verbosity": -1,
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

    # === CLASSIFICATION TARGET ===
    y_train = (y_train_raw >= 0.015).astype(int)
    y_val = (y_val_raw >= 0.015).astype(int)
    # =============================

    if np.sum(y_train) < 20 or np.sum(y_val) < 5:
        print(
            f"  ⚠ Skipping {item_id}: Not enough positive examples in train/val sets (Train: {np.sum(y_train)}, Val: {np.sum(y_val)})."
        )
        return

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print(f"Train spikes: {np.sum(y_train)} / {len(y_train)}")
    print(f"Val spikes: {np.sum(y_val)} / {len(y_val)}")

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
        n_jobs=-1,  # Parralelize trials across all available CPU cores
    )

    params = study.best_params
    if not params:
        print(f"  ⚠ Skipping {item_id}: Optuna could not find any valid parameters.")
        return

    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    params.update(
        {
            "objective": "binary",
            "device_type": "gpu",
            "metric": "binary_logloss",
            "scale_pos_weight": scale_pos_weight,
            "min_data_in_leaf": 50,  # Prevent GPU decision tree split crash
            "max_bin": 63,  # Stable and fast on GPU
            "verbosity": -1,
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

    y_pred_prob = model.predict(X_val_scaled)
    y_pred_binary = (y_pred_prob >= 0.5).astype(int)

    try:
        acc = accuracy_score(y_val, y_pred_binary)
        prec = precision_score(y_val, y_pred_binary, zero_division=0)
        rec = recall_score(y_val, y_pred_binary, zero_division=0)
        f1 = f1_score(y_val, y_pred_binary, zero_division=0)
        auc = roc_auc_score(y_val, y_pred_prob)
    except Exception:
        acc, prec, rec, f1, auc = 0, 0, 0, 0, 0

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f} (When the bot buys, how often is it a real spike?)")
    print(f"Recall:    {rec:.4f} (Out of all real spikes, how many did the bot catch?)")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")

    tested_metrics_dict["accuracy"] = acc
    tested_metrics_dict["precision"] = prec
    tested_metrics_dict["recall"] = rec
    tested_metrics_dict["f1"] = f1
    tested_metrics_dict["roc_auc"] = auc
    tested_metrics_dict["total_validation_samples"] = int(len(y_val))
    tested_metrics_dict["actual_spikes_in_validation"] = int(np.sum(y_val))

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


def analyze_entries(pred_list):
    if not pred_list:
        return []

    now = datetime.now(timezone.utc)
    enriched = []

    for e in pred_list:
        try:
            score = float(e.get("entry_score", 0.0))
        except Exception:
            continue

        # In classification, the score is a probability (0.0 to 1.0)
        # We only want predictions where the model is at least 60% confident
        if score < 0.60:
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

    return enriched


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
