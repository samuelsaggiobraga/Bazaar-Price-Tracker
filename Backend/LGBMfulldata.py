import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import optuna
import json
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import requests
import warnings
from Utils.event_utils import add_skyblock_time_features
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from Utils.data_utils import load_or_fetch_item_data, parse_timestamp
from Utils.mayor_utils import get_mayor_perks, match_mayor_perks
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")



def clean_dump(obj, path):
    """Atomic write with fsync for durability"""
    tmp = path + ".tmp"
    joblib.dump(obj, tmp)
    
    # Force write to disk (Windows-compatible)
    try:
        with open(tmp, "r+b") as f:
            f.flush()
            os.fsync(f.fileno())
    except (OSError, IOError):
        # If fsync fails, continue anyway - file is already written
        pass
    
    # Atomic rename
    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp, path)

def clip_extreme_outliers(y, lower_pct=0.0001, upper_pct=0.9999):
    lower = np.percentile(y, lower_pct*100)
    upper = np.percentile(y, upper_pct*100)
    return np.clip(y, lower, upper)

# =========================================================
# Feature Engineering
# =========================================================

def add_time_features(df, ts_col='timestamp'):
    dt = pd.to_datetime(df[ts_col])
    df['hour'] = dt.dt.hour
    df['minute'] = dt.dt.minute
    df['dayofweek'] = dt.dt.dayofweek

    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['dow_sin'] = np.sin(2*np.pi*df['dayofweek']/7)
    df['dow_cos'] = np.cos(2*np.pi*df['dayofweek']/7)

    df['delta_minutes'] = df['timestamp'].diff().dt.total_seconds().fillna(0) / 60
    return df


def build_lagged_features(
    df,
    price_col='buy_price',
    vol_col='buy_volume',
    lags=(1, 2, 3, 6, 12),
    prefix=''
):
    ret = df[price_col].pct_change()
    df[f'{prefix}ret'] = ret

    for lag in lags:
        df[f'{prefix}ret_lag_{lag}'] = ret.shift(lag)
        df[f'{prefix}price_lag_{lag}'] = df[price_col].shift(lag)
        df[f'{prefix}vol_lag_{lag}'] = df[vol_col].shift(lag)

    df[f'{prefix}roll_mean_6'] = ret.rolling(6).mean()
    df[f'{prefix}roll_std_6'] = ret.rolling(6).std()
    df[f'{prefix}mom_6'] = ret.rolling(6).sum()

    return df

def prepare_dataframe_from_raw(data, mayor_data=None):
    rows = []

    for entry in data:
        try:
            ts = parse_timestamp(entry['timestamp'])
        except:
            continue

        def f(k):
            try:
                return float(entry.get(k, 0))
            except:
                return 0.0

        row = {
            'timestamp': ts,
            'buy_price': f('buy'),
            'sell_price': f('sell'),
            'buy_volume': f('buyVolume'),
            'sell_volume': f('sellVolume'),
            'max_buy': f('maxBuy'),
            'min_buy': f('minBuy'),
        }

        if mayor_data is not None:
            perks = match_mayor_perks(ts, mayor_data)
            for i, v in enumerate(perks):
                row[f'mayor_{i}'] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values('timestamp').reset_index(drop=True)
    df = add_time_features(df)
    df = build_lagged_features(df, price_col='buy_price', vol_col='buy_volume', prefix='buy_')
    df = add_skyblock_time_features(df, ts_col='timestamp')
    return df


# =========================================================
# ENTRY-ONLY LABELING (Regression Target)
# =========================================================

def build_entry_targets(df, horizon_minutes=180, tax=0.0125):
    df = df.copy().sort_values('timestamp').reset_index(drop=True)
    ts = pd.to_datetime(df['timestamp']).astype('int64') // 10**9
    ts = ts.values # Pure numpy for faster indexing
    horizon_sec = horizon_minutes * 60
    
    buy_prices = df['buy_price'].values
    sell_prices = df['sell_price'].values
    n = len(df)
    
    # Pre-allocate all target arrays
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
    profitable_1pct_list = np.zeros(n, dtype=int)
    profitable_2pct_list = np.zeros(n, dtype=int)
    
    # Pre-compute all returns once
    initial_gaps = buy_prices * (1 - tax) - sell_prices
    
    # O(n) sliding window approach
    j = 0
    for i in range(n):
        entry_price = sell_prices[i]
        initial_gap = initial_gaps[i]
        
        # Advance right pointer to end of horizon
        while j < n and ts[j] - ts[i] <= horizon_sec:
            j += 1
        
        # Window is [i, j)
        if j <= i:
            continue
            
        # Compute returns for this window
        returns_horizon = (buy_prices[i:j] * (1 - tax) - entry_price) - initial_gap
        returns_horizon = returns_horizon / (entry_price + 1e-9)
        
        # Expected return and profit probability
        expected_return[i] = np.median(returns_horizon)
        profit_prob[i] = np.mean(returns_horizon > 0)
        
        # Time to first up/down
        up_idxs = np.where(returns_horizon > 0)[0]
        down_idxs = np.where(returns_horizon < 0)[0]
        if len(up_idxs) > 0:
            time_to_first_up[i] = ts[i + up_idxs[0]] - ts[i]
        if len(down_idxs) > 0:
            time_to_first_down[i] = ts[i + down_idxs[0]] - ts[i]
        
        # Max/min and their timing
        t_max_rel = np.argmax(returns_horizon)
        t_min_rel = np.argmin(returns_horizon)
        
        time_to_max[i] = ts[i + t_max_rel] - ts[i]
        time_to_min[i] = ts[i + t_min_rel] - ts[i]
        
        max_profit = returns_horizon[t_max_rel]
        max_loss = returns_horizon[t_min_rel]
        
        max_profit_list[i] = max_profit
        max_loss_list[i] = max_loss
        risk_reward_list[i] = max_profit / abs(max_loss) if max_loss < 0 else max_profit
        
        # Win rates
        win_rate_1pct_list[i] = np.mean(returns_horizon >= 0.01)
        win_rate_2pct_list[i] = np.mean(returns_horizon >= 0.02)
        
        # MAE (Max Adverse Excursion) - worst drawdown before peak
        mae_list[i] = np.min(returns_horizon[:t_max_rel + 1]) if t_max_rel > 0 else 0.0
        
        # MFE (Max Favorable Excursion)
        mfe_list[i] = max_profit
        
        # Profitability flags
        profitable_1pct_list[i] = int(max_profit >= 0.01)
        profitable_2pct_list[i] = int(max_profit >= 0.02)
    
    # Feature engineering (vectorized)
    returns_last_5min = df['buy_price'].pct_change(periods=5)
    returns_last_15min = df['buy_price'].pct_change(periods=15)
    price_vs_5min_high = df['buy_price'] / df['buy_price'].rolling(5).max()
    price_vs_5min_low = df['buy_price'] / df['buy_price'].rolling(5).min()
    price_volatility = df['buy_price'].rolling(20).std() / df['buy_price'].rolling(20).mean()
    spread_volatility = (df['buy_price'] - df['sell_price']).rolling(20).std()
    spread_pct = (df['buy_price'] - df['sell_price']) / df['sell_price']
    spread_momentum = spread_pct.diff()
    
    # Assign all computed values
    df['returns_last_5min'] = returns_last_5min
    df['returns_last_15min'] = returns_last_15min
    df['price_vs_5min_high'] = price_vs_5min_high
    df['price_vs_5min_low'] = price_vs_5min_low
    df['price_volatility'] = price_volatility
    df['spread_volatility'] = spread_volatility
    df['spread_pct'] = spread_pct
    df['spread_momentum'] = spread_momentum
    df['max_profit'] = max_profit_list
    df['max_loss'] = max_loss_list
    df['risk_reward'] = risk_reward_list
    df['win_rate_1pct'] = win_rate_1pct_list
    df['win_rate_2pct'] = win_rate_2pct_list
    df['mae'] = mae_list
    df['mfe'] = mfe_list
    df['profitable_1pct'] = profitable_1pct_list
    df['profitable_2pct'] = profitable_2pct_list
    df['entry_label'] = expected_return
    df['profit_prob'] = profit_prob
    df['time_to_first_up'] = time_to_first_up
    df['time_to_first_down'] = time_to_first_down
    df['time_to_max'] = time_to_max
    df['time_to_min'] = time_to_min
    
    return df

def load_entry_targets(item_id):
    csv_directory = os.path.join(project_root, "csv files")
    df = pd.read_csv(os.path.join(csv_directory, f"{item_id}_debug_data.csv"), parse_dates=['timestamp'])
    return df
# =========================================================
# Cleaning
# =========================================================

def clean_infinite_values(X):
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e8, neginf=-1e8)
    return np.clip(X, -1e8, 1e8)

# =========================================================
# Quantile Loss
# =========================================================

def quantile_loss(y_true, y_pred, alpha):
    diff = y_true - y_pred
    return np.mean(np.maximum(alpha * diff, (alpha - 1) * diff))

# =========================================================
# Optuna Objective (Entry Regression)
# =========================================================

def entry_objective(trial, X, y):
    params = {
        "objective": "regression",
        "learning_rate": trial.suggest_float("lr", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 3,
        "verbosity": -1
    }

    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(params, dtrain, num_boost_round=300)

    preds = model.predict(X)

    # compute sign accuracy
    pred_sign = np.sign(preds)
    true_sign = np.sign(y)
    sign_acc = np.mean(pred_sign == true_sign)

    return sign_acc 


# =========================================================
# Percent Error Stats
# =========================================================

def percent_error_stats(y_true, y_pred, eps=1e-9):
    pct_err = (y_pred - y_true) / (np.abs(y_true) + eps)

    stats = {
        "min_pct_error": np.min(pct_err),
        "max_pct_error": np.max(pct_err),
        "median_pct_error": np.median(pct_err),
        "mean_pct_error": np.mean(pct_err),
        "mean_abs_pct_error": np.mean(np.abs(pct_err)),
        "median_abs_pct_error": np.median(np.abs(pct_err)),
    }

    return stats


# =========================================================
# GENERATE CSV FILES
# =========================================================

def generate_csv_files(item_id):
    mayor_data = get_mayor_perks()
    data = load_or_fetch_item_data(item_id)
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

def train_model_system(item_id, lower = 0.001, upper = 0.999):
    if os.path.exists(os.path.join(project_root, "csv files", f"{item_id}_debug_data.csv")):
        print(f"✓ CSV file for {item_id} already exists")
        df = load_entry_targets(item_id)
    else:
        print(f"✗ CSV file for {item_id} does not exist")
        df = generate_csv_files(item_id)
    
    exclude = {'timestamp', 'entry_label'}
    feature_cols = [c for c in df.columns if c not in exclude]

    X = clean_infinite_values(df[feature_cols].values)
    y = df['entry_label'].values

    # Quick label diagnostics
    try:
        print(
            f"{item_id} labels: min={np.min(y):.6f}, med={np.median(y):.6f}, max={np.max(y):.6f}, "
            f"pos%={(np.mean(y > 0) * 100):.2f}%"
        )
    except Exception:
        pass

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y = clip_extreme_outliers(y, lower_pct=lower, higher_pct=upper)


    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: entry_objective(t, X_scaled, y), n_trials=30)

    params = study.best_params
    params.update({
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1
    })

    model = lgb.train(params, lgb.Dataset(X_scaled, label=y), num_boost_round=400)
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


def test_train_model_system(item_id, lower = 0.001, upper = 0.999):
    if os.path.exists(os.path.join(project_root, "csv files", f"{item_id}_debug_data.csv")):
        print(f"✓ CSV file for {item_id} already exists")
        df = load_entry_targets(item_id)
    else:
        print(f"✗ CSV file for {item_id} does not exist")
        df = generate_csv_files(item_id)

    split_idx = int(len(df) * 0.8)  
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    exclude = {'timestamp', 'entry_label'}
    feature_cols = [c for c in df.columns if c not in exclude]

    X_train = clean_infinite_values(train_df[feature_cols].values)
    y_train = train_df['entry_label'].values
    X_val = clean_infinite_values(val_df[feature_cols].values)
    y_val = val_df['entry_label'].values

    y_val = clip_extreme_outliers(y_val, lower_pct=lower, upper_pct=upper) 
    y_train = clip_extreme_outliers(y_train, lower_pct=lower, upper_pct=upper)


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("y_train: min =", y_train.min(), 
      "median =", np.median(y_train), 
      "max =", y_train.max())
    print("y_val: min =", y_val.min(), 
        "median =", np.median(y_val), 
        "max =", y_val.max())
    pos_train = np.sum(y_train > 0)
    neg_train = np.sum(y_train < 0)
    zero_train = np.sum(y_train == 0)



    print(f"y_train: positive={pos_train}, negative={neg_train}, zero={zero_train}")


    pos_val = np.sum(y_val > 0)
    neg_val = np.sum(y_val < 0)
    zero_val = np.sum(y_val == 0)

    print(f"y_val: positive={pos_val}, negative={neg_val}, zero={zero_val}")
    print(f"y_train: {pos_train/len(y_train)*100:.1f}% positive, {neg_train/len(y_train)*100:.1f}% negative")
    print(f"y_val: {pos_val/len(y_val)*100:.1f}% positive, {neg_val/len(y_val)*100:.1f}% negative")


    print("X_val NaNs:", np.isnan(X_val).sum())
    print("X_val infs:", np.isinf(X_val).sum())


    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: entry_objective(t, X_train_scaled, y_train), n_trials=30)

    params = study.best_params
    params.update({
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1
    })

    model = lgb.train(params, lgb.Dataset(X_train_scaled, label=y_train), num_boost_round=400)

    y_pred = model.predict(X_val_scaled)
    rmse = np.sqrt(np.mean((y_pred - y_val)**2))
    mae = np.mean(np.abs(y_pred - y_val))
    r2 = r2_score(y_val, y_pred)  # compute manually
    y_mean = np.full_like(y_val, y_train.mean())
    baseline_r2 = r2_score(y_val, y_mean)
    print("Baseline R^2:", baseline_r2)
    print(f"RMSE: {rmse}, MAE: {mae}, R^2: {r2}")
    pred_sign = np.sign(model.predict(X_val))
    true_sign = np.sign(y_val)
    accuracy = np.mean(pred_sign == true_sign)
    print("Sign accuracy:", accuracy)
    stats = percent_error_stats(y_val, y_pred)

    for k, v in stats.items():
        print(f"{k}: {v*100:.2f}%")

    mask = y_val > 0.1
    safe_sign_acc = np.mean((y_pred[mask] > 0) == (y_val[mask] > 0))
    print("Safe sign accuracy (true positive returns):", safe_sign_acc)



# =========================================================
# FUTURE PREDICTIONS (MULTI-TIMESTAMP)
# =========================================================



def predict_entries(model, scaler, feature_cols, item_id, mayor_data=None, horizon_hours=3, step_minutes=5):
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=horizon_hours)
    start_str = start.strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")
    end_str = now.strftime("%Y-%m-%dT%H:%M:%S.000").replace(":", "%3A")
    base_url = "https://sky.coflnet.com/api/bazaar"
    url = f"{base_url}/{item_id}/history?start={start_str}&end={end_str}"
    raw = requests.get(url).json()
    df = prepare_dataframe_from_raw(raw, mayor_data)
    if df.empty:
        return []

    last_row = df.iloc[-1:].copy()

    future_times = pd.date_range(start=now, periods=int(horizon_hours*60/step_minutes), freq=f"{step_minutes}min")

    preds = []
    scores = []
    for ts in future_times:
        row = last_row.copy()
        row['timestamp'] = ts

        if mayor_data is not None:
            perks = match_mayor_perks(ts, mayor_data)
            for i, v in enumerate(perks):
                row[f'mayor_{i}'] = v

        for c in feature_cols:
            if c not in row.columns:
                row[c] = 0.0
        row[feature_cols] = row[feature_cols].fillna(0.0)

        X = clean_infinite_values(row[feature_cols].values)
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)[0]
        row['entry_score'] = y_pred
        row['timestamp'] = ts.isoformat()
        scores.append(y_pred)
        preds.append(row[['timestamp', 'buy_price', 'sell_price', 'entry_score']].to_dict(orient='records')[0])

    # Quick prediction diagnostics
    if scores:
        try:
            s = np.array(scores)
            print(
                f"{item_id} scores: min={s.min():.6f}, med={np.median(s):.6f}, max={s.max():.6f}, "
                f"pos%={(np.mean(s > 0) * 100):.2f}%"
            )
        except Exception:
            pass


    return preds


# =========================================================
# ANALYZE TOP PREDICTIONS
# =========================================================

def analyze_entries(pred_list, top_k=5):
    if not pred_list:
        return []

    now = datetime.now(timezone.utc)
    enriched = []

    for e in pred_list:
        try:
            score = float(e.get('entry_score', 0.0))
        except Exception:
            continue

        if score <= 0:
            continue

        ts_str = e.get('timestamp')
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
        enriched_entry['delta_minutes'] = float(delta_minutes)
        enriched.append(enriched_entry)


    enriched.sort(key=lambda x: (x['delta_minutes'], -x['entry_score']))

    return enriched[:top_k]


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    csv_directory = os.path.join(project_root, "csv files")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "bazaar_full_items_ids.json")
    with open(file_path) as f:
        items = json.load(f)

    for entry in items:
        csv_path = os.path.join(csv_directory, f"{entry}_debug_data.csv")
        generate_csv_files(entry)
"""
    for entry in items:
        if entry == "BOOSTER_COOKIE":
            test_train_model_system(entry, lower=0.0001, upper=1.0)
        elif entry == "CONTROL_SWITCH" or entry == "ELECTRON_TRANSMITTER" or entry == "FTX_3070":
            test_train_model_system(entry, lower=0.01, upper=0.99)
        elif entry == "FLAWLESS_SAPPHIRE_GEM":
            test_train_model_system(entry)
"""

