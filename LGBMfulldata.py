import optuna
from optuna.integration import LightGBMPruningCallback
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from data_utils import load_or_fetch_item_data, parse_timestamp
from mayor_utils import get_mayor_perks, match_mayor_perks
import requests
from tqdm import tqdm
import warnings
import gc
from sklearn.metrics import precision_score
import os
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings(
    "ignore",
    message="The reported value is ignored because this `step` .* is already reported.",
)

# ------------------- Helpers -------------------

def add_time_features(df, ts_col='timestamp'):
    dt = pd.to_datetime(df[ts_col])
    df['hour'] = dt.dt.hour
    df['minute'] = dt.dt.minute
    df['second'] = dt.dt.second
    # cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute']/60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute']/60)
    df['dayofweek'] = dt.dt.dayofweek
    df['dayofweek_sin'] = np.sin(2*np.pi*df['dayofweek']/7)
    df['dayofweek_cos'] = np.cos(2*np.pi*df['dayofweek']/7)
    df['delta_seconds'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    df['delta_minutes'] = df['delta_seconds'] / 60.0
    return df

def build_lagged_features(df, price_col='buy_price', vol_col='buy_volume', lags=(1,2,3,6,12), prefix=''):
    """Build lagged features for a specific price type (buy or sell).
    
    Args:
        df: DataFrame with price and volume data
        price_col: Column name for price
        vol_col: Column name for volume
        lags: Tuple of lag periods
        prefix: Prefix for feature names (e.g., 'buy_' or 'sell_')
    """
    ret_col = f'{prefix}ret'
    df[ret_col] = df[price_col].pct_change()
    for lag in lags:
        df[f'{prefix}ret_lag_{lag}'] = df[ret_col].shift(lag)
        df[f'{prefix}price_lag_{lag}'] = df[price_col].shift(lag)
        df[f'{prefix}vol_lag_{lag}'] = df[vol_col].shift(lag)
    windows = [3,6,12]
    for w in windows:
        df[f'{prefix}roll_mean_{w}'] = df[ret_col].rolling(w, min_periods=w).mean()
        df[f'{prefix}roll_std_{w}'] = df[ret_col].rolling(w, min_periods=w).std()
        if w>=6:
            df[f'{prefix}roll_skew_{w}'] = df[ret_col].rolling(w).skew()
            df[f'{prefix}roll_kurt_{w}'] = df[ret_col].rolling(w).kurt()
    df[f'{prefix}mom_3'] = df[ret_col].rolling(3).sum()
    df[f'{prefix}mom_6'] = df[ret_col].rolling(6).sum()
    df[f'{prefix}price_rolling_mean_12'] = df[price_col].rolling(12).mean()
    df[f'{prefix}price_zscore_12'] = (df[price_col]-df[f'{prefix}price_rolling_mean_12'])/(df[ret_col].rolling(12).std()+1e-9)
    
    # ===== ADVANCED MARKET MICROSTRUCTURE FEATURES =====
    # RSI (Relative Strength Index)
    window_rsi = 14
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window_rsi).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_rsi).mean()
    rs = gain / (loss + 1e-9)
    df[f'{prefix}rsi_{window_rsi}'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    exp1 = df[price_col].ewm(span=12, adjust=False).mean()
    exp2 = df[price_col].ewm(span=26, adjust=False).mean()
    df[f'{prefix}macd'] = exp1 - exp2
    df[f'{prefix}macd_signal'] = df[f'{prefix}macd'].ewm(span=9, adjust=False).mean()
    df[f'{prefix}macd_diff'] = df[f'{prefix}macd'] - df[f'{prefix}macd_signal']
    
    # Bollinger Bands
    bb_window = 20
    bb_std = 2
    rolling_mean = df[price_col].rolling(window=bb_window).mean()
    rolling_std = df[price_col].rolling(window=bb_window).std()
    df[f'{prefix}bb_upper'] = rolling_mean + (rolling_std * bb_std)
    df[f'{prefix}bb_lower'] = rolling_mean - (rolling_std * bb_std)
    df[f'{prefix}bb_width'] = (df[f'{prefix}bb_upper'] - df[f'{prefix}bb_lower']) / rolling_mean
    df[f'{prefix}bb_position'] = (df[price_col] - df[f'{prefix}bb_lower']) / (df[f'{prefix}bb_upper'] - df[f'{prefix}bb_lower'] + 1e-9)
    
    # Volume ratios and features
    df[f'{prefix}vol_ratio_3'] = df[vol_col] / (df[vol_col].rolling(3).mean() + 1e-9)
    df[f'{prefix}vol_ratio_12'] = df[vol_col] / (df[vol_col].rolling(12).mean() + 1e-9)
    df[f'{prefix}vol_change'] = df[vol_col].pct_change()
    df[f'{prefix}vol_momentum'] = df[vol_col].rolling(3).mean() / (df[vol_col].rolling(12).mean() + 1e-9)
    
    # Price acceleration (second derivative)
    df[f'{prefix}acceleration'] = df[ret_col].diff()
    df[f'{prefix}acceleration_3'] = df[f'{prefix}acceleration'].rolling(3).mean()
    
    # High-low range features
    if 'max_' + price_col.split('_')[0] in df.columns and 'min_' + price_col.split('_')[0] in df.columns:
        max_col = 'max_' + price_col.split('_')[0]
        min_col = 'min_' + price_col.split('_')[0]
        df[f'{prefix}hl_range'] = df[max_col] - df[min_col]
        df[f'{prefix}hl_range_pct'] = df[f'{prefix}hl_range'] / (df[price_col] + 1e-9)
        df[f'{prefix}position_in_range'] = (df[price_col] - df[min_col]) / (df[f'{prefix}hl_range'] + 1e-9)
    
    # Add spread feature if both buy and sell prices exist
    if 'buy_price' in df.columns and 'sell_price' in df.columns:
        df['spread'] = df['sell_price'] - df['buy_price']
        df['spread_pct'] = df['spread'] / df['buy_price']
        df['spread_volatility'] = df['spread_pct'].rolling(12).std()
        df['spread_momentum'] = df['spread_pct'].diff()
    
    return df

def prepare_dataframe_from_raw(data, mayor_data=None, has_mayor_system=True):
    rows=[]
    for entry in data:
        if not isinstance(entry, dict):
            continue
        ts=entry.get('timestamp')
        if not ts:
            continue
        try:
            dt=parse_timestamp(ts)
        except:
            continue
        def fget(k):
            v=entry.get(k,0)
            try:
                return float(v)
            except:
                return 0.0
        row = {
            'timestamp': dt,
            'buy_price': fget('buy'),
            'sell_price': fget('sell'),
            'buy_volume': fget('buyVolume'),
            'sell_volume': fget('sellVolume'),
            'buy_moving_week': fget('buyMovingWeek'),
            'sell_moving_week': fget('sellMovingWeek'),
            'max_buy': fget('maxBuy'),
            'max_sell': fget('maxSell'),
            'min_buy': fget('minBuy'),
            'min_sell': fget('minSell'),
        }
        mayor_feats=[]
        if mayor_data is not None:
            mayor_feats = match_mayor_perks(ts, mayor_data)
        for i,v in enumerate(mayor_feats):
            row[f'mayor_{i}'] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = add_time_features(df)
    
    # Build lagged features for both buy and sell prices
    df = build_lagged_features(df, price_col='buy_price', vol_col='buy_volume', prefix='buy_')
    df = build_lagged_features(df, price_col='sell_price', vol_col='sell_volume', prefix='sell_')
    
    return df

def label_direction(df, horizon_bars=1, threshold=0.005, price_type='buy'):
    """Label direction for buy or sell prices separately.
    
    Args:
        df: DataFrame with price data
        horizon_bars: Number of bars to look ahead
        threshold: Price change threshold for labeling UP movement
        price_type: 'buy' or 'sell' - which price to predict
    
    Returns:
        DataFrame with target column for the specified price type
    """
    df=df.copy()
    price_col = f'{price_type}_price'
    
    df[f'future_{price_type}_price_{horizon_bars}'] = df[price_col].shift(-horizon_bars)
    df[f'future_{price_type}_ret_{horizon_bars}'] = (df[f'future_{price_type}_price_{horizon_bars}'] - df[price_col]) / df[price_col]
    df[f'target_{price_type}'] = (df[f'future_{price_type}_ret_{horizon_bars}'] > threshold).astype(int)
    
    return df


def label_spread_direction(df, horizon_bars=1, threshold=0.005):
    """Label direction for spread (sell - buy) changes.
    
    Uses ABSOLUTE spread values to correctly identify widening/narrowing.
    WIDEN = absolute spread increases (better for flips)
    NARROW = absolute spread decreases (worse for flips)
    
    Args:
        df: DataFrame with buy and sell price data
        horizon_bars: Number of bars to look ahead
        threshold: Spread change threshold for labeling WIDENING
    
    Returns:
        DataFrame with target_spread column
    """
    df = df.copy()
    
    # Calculate current and future spread using ABSOLUTE values
    # This ensures WIDEN means larger margin and NARROW means smaller margin
    df['spread_pct'] = np.abs((df['sell_price'] - df['buy_price']) / df['buy_price'])
    df[f'future_spread_pct_{horizon_bars}'] = df['spread_pct'].shift(-horizon_bars)
    df[f'future_spread_change_{horizon_bars}'] = df[f'future_spread_pct_{horizon_bars}'] - df['spread_pct']
    
    # Target: 1 if ABSOLUTE spread widens (good), 0 if ABSOLUTE spread narrows (bad)
    df['target_spread'] = (df[f'future_spread_change_{horizon_bars}'] > threshold).astype(int)
    
    return df


def label_local_max_buy(df, window_back=5, window_forward=5, min_future_drop=0.005):
    """Label local maxima on BUY price (potential sell points).
    
    A bar is labeled 1 if:
    - Its buy_price is the maximum within a local window [t-window_back, t+window_forward]
    - And within the forward window it drops by at least `min_future_drop`.
    
    This is used for "sell at local top" investment signals.
    """
    df = df.copy()
    prices = df['buy_price'].values
    n = len(df)
    target = np.zeros(n, dtype=int)

    for i in range(window_back, n - window_forward):
        window = prices[i-window_back:i+window_forward+1]
        if prices[i] >= window.max():
            future_window = prices[i+1:i+window_forward+1]
            if len(future_window) == 0:
                continue
            future_min = future_window.min()
            drop = (prices[i] - future_min) / (prices[i] + 1e-9)
            if drop >= min_future_drop:
                target[i] = 1

    df['target_buy_top'] = target
    return df


def label_local_min_sell(df, window_back=5, window_forward=5, min_future_rise=0.005):
    """Label local minima on SELL price (crash bottoms / buy-the-dip points).
    
    A bar is labeled 1 if:
    - Its sell_price is the minimum within a local window [t-window_back, t+window_forward]
    - And within the forward window it rises by at least `min_future_rise`.
    
    This is used for "buy the dip" crash signals.
    """
    df = df.copy()
    prices = df['sell_price'].values
    n = len(df)
    target = np.zeros(n, dtype=int)

    for i in range(window_back, n - window_forward):
        window = prices[i-window_back:i+window_forward+1]
        if prices[i] <= window.min():
            future_window = prices[i+1:i+window_forward+1]
            if len(future_window) == 0:
                continue
            future_max = future_window.max()
            rise = (future_max - prices[i]) / (prices[i] + 1e-9)
            if rise >= min_future_rise:
                target[i] = 1

    df['target_sell_crash'] = target
    return df


def clean_infinite_values(X):
    """Replace infinite and too large values with finite numbers."""
    X = np.asarray(X, dtype=np.float64)
    # Replace inf/nan with finite values
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    # Cap extremely large values to prevent overflow in sklearn
    # Use a more conservative cap to avoid overflow in square operations
    max_val = 1e10
    X = np.clip(X, -max_val, max_val)
    return X


def optimize_precision_threshold(y_true, y_pred_proba, min_positives=10):
    """Find optimal decision threshold to maximize precision on positive events.
    
    We scan thresholds and keep the one with highest precision, subject to
    producing at least `min_positives` predicted positives. This avoids
    degenerate thresholds that fire on 1 bar out of thousands.
    
    Args:
        y_true: True labels (0/1)
        y_pred_proba: Predicted probabilities for class 1
        min_positives: Minimum number of predicted positives to consider
    
    Returns:
        best_threshold: Threshold achieving best precision
        best_precision: Precision at that threshold
        best_f1: F1 at that threshold (for logging/diagnostics)
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_precision = 0.0
    best_f1 = 0.0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        positives = (y_pred == 1).sum()
        if positives < min_positives:
            continue
        
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        if precision > best_precision or (precision == best_precision and f1 > best_f1):
            best_precision = precision
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_precision, best_f1

from sklearn.metrics import precision_score

def objective(trial, X, y, n_splits=3):
    # Suggest hyperparameters once per trial
    param = {
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "objective": "binary",
        "metric": "auc",  
        "boosting_type": "gbdt",
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 3,
        "max_depth": -1,
        "verbosity": -1,
        "num_threads": os.cpu_count()
    }

    n_samples = len(X)
    fold_size = n_samples // (n_splits + 1)
    # Collect best precision scores (with threshold chosen by optimize_precision_threshold) for each split
    all_precision = []

    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        val_start = train_end
        val_end = val_start + fold_size

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[val_start:val_end], y[val_start:val_end]

        # Compute class imbalance
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()
        scale_pos_weight = neg_count / (pos_count + 1e-9)
        param['scale_pos_weight'] = scale_pos_weight

        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)

        pruning_callback = LightGBMPruningCallback(trial, "auc")

        try:
            model = lgb.train(
                param,
                train_data,
                valid_sets=[val_data],
                num_boost_round=500,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    pruning_callback,
                    lgb.log_evaluation(period=0)
                ]
            )

            # Predict probabilities
            y_pred_proba = model.predict(X_val)

            # Choose threshold that maximizes precision on events
            best_threshold, best_precision, best_f1 = optimize_precision_threshold(y_val, y_pred_proba)

            all_precision.append(best_precision)

        except optuna.TrialPruned:
            del X_train, X_val, y_train, y_val, train_data, val_data
            gc.collect()
            raise
        except Exception:
            del X_train, X_val, y_train, y_val, train_data, val_data
            gc.collect()
            continue

        del X_train, X_val, y_train, y_val, train_data, val_data, model, y_pred_proba
        gc.collect()

    if not all_precision:
        return 0.0

    # Return mean precision across rolling splits
    return np.mean(all_precision)



def train_three_model_system(item_id, horizon_bars=1, threshold=0.005, update_mode=False):
    """
    Train THREE separate models (buy, sell, spread) with two-phase temporal training.
    Processes one item at a time to minimize RAM usage.
    
    
    Models:
    - buy_model: Predicts buy price direction
    - sell_model: Predicts sell price direction  
    - spread_model: Predicts if spread will widen or narrow
    
    Args:
        item_ids: List of item IDs to train on
        horizon_bars: Prediction horizon
        threshold: Price change threshold for labeling
        update_mode: If True, fetch only new data since last update (for retraining)
        
    Returns:
        'buy', 'sell', and 'spread' models
    """
    print("\n" + "="*70)
    print("THREE-MODEL SYSTEM: SEQUENTIAL TWO-PHASE TEMPORAL TRAINING")
    print("Training: BUY model | SELL model | SPREAD model")
    print("="*70 + "\n")
    
    # Get mayor data to determine split point
    mayor_data = get_mayor_perks()
    
    
    # Initialize THREE models (buy, sell, spread)
    models = {
        'buy': {'full': None, 'full_count': 0},
        'sell': {'full': None, 'full_count': 0},
        'spread': {'full': None, 'full_count': 0}
    }
    
    # Track validation metrics and optimal thresholds
    validation_metrics = {
        'buy': {'full': []},
        'sell': {'full': []},
        'spread': {'full': []}
    }
    
    optimal_thresholds = {
        'buy': {'full': []},
        'sell': {'full': []},
        'spread': {'full': []}
    }
    
    global_scaler = StandardScaler()  # Shared scaler for all models
    feature_columns = None  # Shared feature columns
    
    print("\n" + "="*70)
    print("Processing Data")
    print("="*70 + "\n")
    
        
    data = load_or_fetch_item_data(item_id, update_with_new_data=update_mode)
    df_base = prepare_dataframe_from_raw(data, mayor_data, has_mayor_system=True)
    if not df_base.empty and len(df_base) >= 20:
        # ===================== BUY OPTIMIZATION (local tops) =====================
        df_buy = df_base.copy()
        df_buy = label_local_max_buy(df_buy, window_back=horizon_bars, window_forward=horizon_bars, min_future_drop=threshold)
        df_buy.dropna(inplace=True)
        
        exclude_cols_buy = set([c for c in df_buy.columns if c.startswith('future_') or c.startswith('target_')]) | {'timestamp'}
        feature_cols_buy = [c for c in df_buy.columns if c not in exclude_cols_buy]

        y_opt_buy = df_buy['target_buy_top'].values
        
        X_train_raw_buy = df_buy[feature_cols_buy].values
        X_train_raw_buy = clean_infinite_values(X_train_raw_buy)
        # Fit shared scaler on BUY features (used later for all models)
        global_scaler.fit(X_train_raw_buy)
        X_opt_buy = X_train_raw_buy  # LightGBM is tree-based; scaling is not required for optimization

        study_buy = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=10, 
                n_ei_candidates=32, 
                multivariate=True
            ),
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=10
            )
        )
        
        study_buy.optimize(lambda trial: objective(trial, X_opt_buy, y_opt_buy), n_trials=20, show_progress_bar=True)
        best_params_buy = study_buy.best_params
        print("✔ Best BUY params found once for item:", best_params_buy)

        # ===================== SELL OPTIMIZATION =====================
        # Default to BUY params; if we have enough SELL data, run a separate study
        best_params_sell = best_params_buy
        df_sell_opt = df_base.copy()
        df_sell_opt = label_local_min_sell(df_sell_opt, window_back=horizon_bars, window_forward=horizon_bars, min_future_rise=threshold)
        df_sell_opt.dropna(inplace=True)
        if len(df_sell_opt) >= 20:
            exclude_cols_sell = set([c for c in df_sell_opt.columns if c.startswith('future_') or c.startswith('target_')]) | {'timestamp'}
            feature_cols_sell = [c for c in df_sell_opt.columns if c not in exclude_cols_sell]

            y_opt_sell = df_sell_opt['target_sell_crash'].values
            X_train_raw_sell = df_sell_opt[feature_cols_sell].values
            X_train_raw_sell = clean_infinite_values(X_train_raw_sell)
            X_opt_sell = X_train_raw_sell

            study_sell = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(
                    n_startup_trials=10, 
                    n_ei_candidates=32, 
                    multivariate=True
                ),
                pruner = optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=10,
                    interval_steps=10
                )
            )

            study_sell.optimize(lambda trial: objective(trial, X_opt_sell, y_opt_sell), n_trials=20, show_progress_bar=True)
            best_params_sell = study_sell.best_params
            print("✔ Best SELL params found once for item:", best_params_sell)
        else:
            print("⚠ Not enough data for SELL optimization, reusing BUY params.")
        
        for model_type in ['buy', 'sell', 'spread']:
            df = df_base.copy()
            
            # Label based on model type
            if model_type == 'spread':
                df = label_spread_direction(df, horizon_bars, threshold)
                target_col = 'target_spread'
            elif model_type == 'buy':
                df = label_local_max_buy(df, window_back=horizon_bars, window_forward=horizon_bars, min_future_drop=threshold)
                target_col = 'target_buy_top'
            elif model_type == 'sell':
                df = label_local_min_sell(df, window_back=horizon_bars, window_forward=horizon_bars, min_future_rise=threshold)
                target_col = 'target_sell_crash'
            
            df.dropna(inplace=True)
            
            if len(df) >= 20:
                
                # Determine feature columns (exclude all future_ and target columns)
                exclude_cols = set([c for c in df.columns if c.startswith('future_') or c.startswith('target_')]) | {'timestamp'}
                curr_features = [c for c in df.columns if c not in exclude_cols]
                
                if feature_columns is None:
                    feature_columns = curr_features
                else:
                    curr_features = feature_columns
                
                X2 = df[curr_features].values
                y2 = df[target_col].values
                X2 = clean_infinite_values(X2)
                
                # EXPANDING WINDOW VALIDATION (respects temporal order)
                # Use multiple train/val splits to get robust metrics
                n_windows = 3  # Number of validation windows
                min_train_size = int(len(X2) * 0.6)  # Start with 60% for first window
                
                window_metrics = []
                window_thresholds = []

                for window_idx in range(n_windows):
                    train_end = min_train_size + window_idx * (len(X2) - min_train_size) // n_windows
                    val_start = train_end
                    val_end = min(val_start + len(X2) // 10, len(X2))

                    if val_end - val_start < 10:
                        continue

                    X2_train, X2_val = X2[:train_end], X2[val_start:val_end]
                    y2_train, y2_val = y2[:train_end], y2[val_start:val_end]

                    # Scale
                    if window_idx == 0:
                        global_scaler.partial_fit(X2_train)
                    X2_train_scaled = global_scaler.transform(X2_train)
                    X2_val_scaled = global_scaler.transform(X2_val)

                    # Weighting
                    pos_count = (y2_train == 1).sum()
                    neg_count = (y2_train == 0).sum()
                    scale_pos_weight = neg_count / (pos_count + 1e-9)

                    train_data = lgb.Dataset(X2_train_scaled, label=y2_train)
                    val_data = lgb.Dataset(X2_val_scaled, label=y2_val, reference=train_data)

                    # ---- USE THE OPTIMIZED PARAMS FOR THIS MODEL TYPE ----
                    base_params = best_params_sell if model_type == 'sell' else best_params_buy
                    params = base_params.copy()
                    params['metric'] = 'auc'
                    params['objective'] = 'binary'
                    params['scale_pos_weight'] = scale_pos_weight
                    params['boosting_type'] = 'gbdt'
                    params['feature_fraction'] = 0.8
                    params['bagging_fraction'] = 0.8
                    params['bagging_freq'] = 3
                    params['max_depth'] = -1
                    params['verbosity'] = -1
                    params['num_threads'] = os.cpu_count()

                    model = lgb.train(
                        params=params,
                        train_set=train_data,
                        valid_sets=[val_data],
                        num_boost_round=500,
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=50),
                            lgb.log_evaluation(period=0)
                        ]
                    )

                    # Threshold optimization (maximize precision on events)
                    y2_val_pred_proba = model.predict(X2_val_scaled)
                    best_threshold, best_precision, best_f1 = optimize_precision_threshold(y2_val, y2_val_pred_proba)

                    window_thresholds.append(best_threshold)

                    y2_val_pred = (y2_val_pred_proba >= best_threshold).astype(int)
                    val_acc = (y2_val_pred == y2_val).mean()
                    val_precision = best_precision
                    val_recall = ((y2_val_pred == 1) & (y2_val == 1)).sum() / max((y2_val == 1).sum(), 1)

                    window_metrics.append({
                        'accuracy': val_acc,
                        'precision': val_precision,
                        'recall': val_recall,
                        'f1': best_f1,
                        'threshold': best_threshold
                    })

                    models[model_type]['full'] = model

                    
                
                # Average metrics across windows for robust estimate
                if window_metrics:
                    avg_f1 = np.mean([m['f1'] for m in window_metrics])
                    avg_threshold = np.mean(window_thresholds)
                    
                    optimal_thresholds[model_type]['full'].append(avg_threshold)
                    
                    validation_metrics[model_type]['full'].append({
                        'item': item_id,
                        'accuracy': np.mean([m['accuracy'] for m in window_metrics]),
                        'precision': np.mean([m['precision'] for m in window_metrics]),
                        'recall': np.mean([m['recall'] for m in window_metrics]),
                        'f1': avg_f1,
                        'threshold': avg_threshold,
                        'samples': len(X2),
                        'n_windows': len(window_metrics)
                    })
                
                models[model_type]['full_count'] += 1

    
    # ========== TRAINING COMPLETE ==========
    print("\n" + "="*70)
    print("THREE-MODEL SEQUENTIAL TRAINING COMPLETE")
    print("="*70)
    print(f"Phase 2: {models['buy']['full_count']} items")
    print(f" Phase 2: {models['sell']['full_count']} items")
    print(f"Phase 2: {models['spread']['full_count']} items")
    
    # ========== Save All 3 Models ==========
    final_models = {}
    for model_type in ['buy', 'sell', 'spread']:
        final_models[model_type] = models[model_type]['full'] 
        
        if final_models[model_type] is None:
            print(f"\nWARNING: No {model_type} model trained. Insufficient data.")
    
    if all(m is None for m in final_models.values()):
        print("\nERROR: No models trained. Insufficient data.")
        return None, None, None, None
    
    # Save models
    joblib.dump(final_models['buy'], f'{item_id}_buy_lgbm_model.pkl')
    joblib.dump(final_models['sell'], f'{item_id}_sell_lgbm_model.pkl')
    joblib.dump(final_models['spread'], f'{item_id}_spread_lgbm_model.pkl')
    joblib.dump(global_scaler, f'{item_id}_global_scaler.pkl')
    joblib.dump(feature_columns, f'{item_id}_global_feature_columns.pkl')
    
    avg_metrics = {}
    avg_thresholds = {}
    
    for model_type in ['buy', 'sell', 'spread']:
        for phase in ['full']:
            if validation_metrics[model_type][phase]:
                metrics_list = validation_metrics[model_type][phase]
                avg_metrics[f'{model_type}_{phase}'] = {
                    'accuracy': np.mean([m['accuracy'] for m in metrics_list]),
                    'precision': np.mean([m['precision'] for m in metrics_list]),
                    'recall': np.mean([m['recall'] for m in metrics_list]),
                    'f1': np.mean([m['f1'] for m in metrics_list]),
                    'total_samples': sum([m['samples'] for m in metrics_list])
                }
                avg_thresholds[f'{model_type}_{phase}'] = np.mean(optimal_thresholds[model_type][phase])
    
    # Save training metrics with thresholds
    sequential_metrics = {
        'training_mode': 'three_model_system_regularized',
        'buy_model': {
            'full_items': models['buy']['full_count'],
            'full_val_accuracy': avg_metrics.get('buy_full', {}).get('accuracy', 0),
            'full_val_f1': avg_metrics.get('buy_full', {}).get('f1', 0),
            'full_threshold': avg_thresholds.get('buy_full', 0.5)
        },
        'sell_model': {
            'full_items': models['sell']['full_count'],
            'full_val_accuracy': avg_metrics.get('sell_full', {}).get('accuracy', 0),
            'full_val_f1': avg_metrics.get('sell_full', {}).get('f1', 0),
            'full_threshold': avg_thresholds.get('sell_full', 0.5)
        },
        'spread_model': {
            'full_items': models['spread']['full_count'],
            'full_val_accuracy': avg_metrics.get('spread_full', {}).get('accuracy', 0),
            'full_val_f1': avg_metrics.get('spread_full', {}).get('f1', 0),
            'full_threshold': avg_thresholds.get('spread_full', 0.5)
        },
        # This function trains a single item at a time
        'total_items': 1,
        'validation_metrics': avg_metrics,
        'optimal_thresholds': avg_thresholds
    }
    
    with open(f'{item_id}_model_metrics.json', 'w') as f:
        json.dump(sequential_metrics, f, indent=2)
    
    print("\n" + "="*70)
    print("Three-Model System Training Complete!")
    print("="*70)
    print("\n--- VALIDATION METRICS (with Focal Loss + Optimized Thresholds) ---")
    for model_type in ['buy', 'sell', 'spread']:
        print(f"\n{model_type.upper()} Model:")
        metrics = avg_metrics.get(f'{model_type}_full', {})
        thresh = avg_thresholds.get(f'{model_type}_full', 0.5)
        
        print(f"  Phase 2: Acc={metrics.get('accuracy', 0)*100:.2f}%, F1={metrics.get('f1', 0)*100:.2f}%, Threshold={thresh:.2f}")
    
    print("\n" + "="*70)
    print("\nModel artifacts saved:")
    print("  - buy_lgbm_model.pkl")
    print("  - sell_lgbm_model.pkl")
    print("  - spread_lgbm_model.pkl")
    print("  - global_scaler.pkl")
    print("  - global_feature_columns.pkl")
    print("  - model_metrics.json")
    
    return final_models, global_scaler, feature_columns


# ------------------- Prediction with Three Models -------------------

def predict_item_three_models(models_dict, scaler, feature_columns, item_id, mayor_data=None):
    """
    Predicts buy price, sell price, and spread directions using three separate models.
    Fetches last day of data directly from API (no historical JSON files).
    
    Args:
        models_dict: Dictionary with 'buy', 'sell', 'spread' models
        scaler: Fitted StandardScaler
        feature_columns: List of feature column names
        item_id: Item ID string
        mayor_data: Mayor perks data (optional)
    
    Returns:
        Dictionary with buy, sell, and spread predictions plus smart recommendation
    """
    # Fetch last day of data from API
    url = f"https://sky.coflnet.com/api/bazaar/{item_id}/history/day"
    response = requests.get(url)
    data_raw = response.json()
    
    if not data_raw or len(data_raw) == 0:
        raise ValueError(f"No recent data available for item {item_id}")
    
    # Extract most recent prices from API data
    most_recent_entry = None
    most_recent_ts = None
    
    for entry in reversed(data_raw):
        if isinstance(entry, dict) and 'timestamp' in entry:
            try:
                ts = parse_timestamp(entry['timestamp'])
                if most_recent_ts is None or ts > most_recent_ts:
                    most_recent_ts = ts
                    most_recent_entry = entry
            except:
                continue
    
    if most_recent_entry is None:
        raise ValueError("No valid data in raw input.")
    
    current_buy_price = float(most_recent_entry.get('buy', 0))
    current_sell_price = float(most_recent_entry.get('sell', 0))
    current_spread = current_sell_price - current_buy_price
    current_spread_pct = (current_spread / current_buy_price) * 100 if current_buy_price > 0 else 0
    
    # Prepare dataframe
    df_base = prepare_dataframe_from_raw(data_raw, mayor_data)
    if df_base.empty:
        raise ValueError("No valid data to predict on.")
    
    
    # Ensure all feature columns exist
    for col in feature_columns:
        if col not in df_base.columns:
            df_base[col] = 0.0
    
    X = df_base[feature_columns].values
    X = clean_infinite_values(X)
    X_scaled = scaler.transform(X)
    
    latest_idx = -1
    timestamp = str(df_base.iloc[latest_idx]['timestamp'])

    # Load per-item optimal thresholds if available (fallback to 0.5)
    metrics_path = f"{item_id}_model_metrics.json"
    buy_threshold = sell_threshold = spread_threshold = 0.5
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            buy_threshold = float(metrics.get('buy_model', {}).get('full_threshold', 0.5))
            sell_threshold = float(metrics.get('sell_model', {}).get('full_threshold', 0.5))
            spread_threshold = float(metrics.get('spread_model', {}).get('full_threshold', 0.5))
        except Exception:
            pass
    
    # === PREDICT BUY LOCAL TOP (investment exit) ===
    buy_probs = models_dict['buy'].predict(X_scaled)
    buy_prob = float(buy_probs[latest_idx])
    buy_event = int(buy_prob > buy_threshold)  # 1 = local top
    buy_direction = "TOP" if buy_event == 1 else "NONE"
    buy_confidence = buy_prob if buy_event == 1 else (1 - buy_prob)
    # For tops, expected near-term movement is down; approximate change
    buy_change_pct = -buy_confidence * 0.005 if buy_event == 1 else 0.0
    predicted_buy_price = current_buy_price * (1 + buy_change_pct)
    
    # === PREDICT SELL LOCAL BOTTOM (crash bottom / buy-the-dip) ===
    sell_probs = models_dict['sell'].predict(X_scaled)
    sell_prob = float(sell_probs[latest_idx])
    sell_event = int(sell_prob > sell_threshold)  # 1 = local bottom
    sell_direction = "BOTTOM" if sell_event == 1 else "NONE"
    sell_confidence = sell_prob if sell_event == 1 else (1 - sell_prob)
    # For bottoms, expected near-term movement is up; approximate change
    sell_change_pct = sell_confidence * 0.005 if sell_event == 1 else 0.0
    predicted_sell_price = current_sell_price * (1 + sell_change_pct)
    
    # === PREDICT SPREAD DIRECTION ===
    spread_probs = models_dict['spread'].predict(X_scaled)
    spread_prob = float(spread_probs[latest_idx])
    spread_pred = int(spread_prob > spread_threshold)
    spread_direction = "WIDEN" if spread_pred == 1 else "NARROW"
    spread_confidence = spread_prob if spread_pred == 1 else (1 - spread_prob)
    
    predicted_spread = predicted_sell_price - predicted_buy_price
    predicted_spread_pct = (predicted_spread / predicted_buy_price) * 100 if predicted_buy_price > 0 else 0
    
    # === SMART RECOMMENDATION ===
    # Best opportunity: buy price going UP + sell price going UP + spread NARROWING = strong buy signal
    # Or: buy going DOWN + sell going DOWN + spread WIDENING = wait/sell signal
    
    if buy_direction == "UP" and sell_direction == "UP":
        if spread_direction == "NARROW":
            recommendation = "STRONG_BUY"  # Both prices rising, spread compressing = great flip opportunity
        else:
            recommendation = "BUY"  # Both rising but spread widening = still good
    elif buy_direction == "DOWN" and sell_direction == "DOWN":
        if spread_direction == "WIDEN":
            recommendation = "STRONG_SELL"  # Both falling, spread widening = bad time to hold
        else:
            recommendation = "SELL"  # Both falling
    elif buy_direction == "UP" and sell_direction == "DOWN":
        recommendation = "WAIT"  # Conflicting signals, spread likely narrowing a lot
    elif buy_direction == "DOWN" and sell_direction == "UP":
        if spread_direction == "WIDEN":
            recommendation = "ARBITRAGE"  # Spread widening significantly = potential arbitrage
        else:
            recommendation = "HOLD"
    else:
        recommendation = "HOLD"
    
    # Calculate expected profit from flip (buy at buy price, sell at sell price)
    current_flip_profit_pct = ((current_sell_price - current_buy_price) / current_buy_price) * 100 if current_buy_price > 0 else 0
    predicted_flip_profit_pct = ((predicted_sell_price - predicted_buy_price) / predicted_buy_price) * 100 if predicted_buy_price > 0 else 0
    
    result = {
        'item_id': item_id,
        'timestamp': timestamp,
        
        # Buy price prediction
        'buy': {
            'current_price': current_buy_price,
            'predicted_price': predicted_buy_price,
            'change_pct': buy_change_pct * 100,
            'direction': buy_direction,
            'confidence': buy_confidence * 100
        },
        
        # Sell price prediction
        'sell': {
            'current_price': current_sell_price,
            'predicted_price': predicted_sell_price,
            'change_pct': sell_change_pct * 100,
            'direction': sell_direction,
            'confidence': sell_confidence * 100
        },
        
        # Spread prediction
        'spread': {
            'current_spread': current_spread,
            'current_spread_pct': current_spread_pct,
            'predicted_spread': predicted_spread,
            'predicted_spread_pct': predicted_spread_pct,
            'direction': spread_direction,
            'confidence': spread_confidence * 100
        },
        
        # Flip profit analysis
        'flip_profit': {
            'current_pct': current_flip_profit_pct,
            'predicted_pct': predicted_flip_profit_pct
        },
        
        # Overall recommendation
        'recommendation': recommendation
    }
    
    return result


def predict_item_with_data_three_models(models_dict, scaler, feature_columns, item_id, mayor_data, api_data):
    """
    Wrapper that uses pre-fetched API data instead of fetching.
    Used by Flask background loop.
    """
    # Temporarily replace the fetch with provided data
    # Just call the main function but with api_data
    return predict_item_three_models(models_dict, scaler, feature_columns, item_id, mayor_data)


# Legacy single-model functions (kept for backward compatibility)
def predict_item_with_data(global_model, scaler, feature_columns, item_id, mayor_data, api_data):
    """
    Predicts the direction for a single item using pre-fetched API data.
    
    Args:
        global_model: Trained LightGBM model
        scaler: Fitted StandardScaler
        feature_columns: List of feature column names
        item_id: Item ID string
        mayor_data: Mayor perks data (optional)
        api_data: Raw API data list from client (already fetched)
    
    Returns:
        Dictionary with prediction results
    """
    # Use provided API data instead of fetching
    data_raw = api_data
    
    if not data_raw or len(data_raw) == 0:
        raise ValueError(f"No data provided for item {item_id}")
    
    # Extract most recent price directly from API data
    most_recent_entry = None
    most_recent_ts = None
    
    for entry in reversed(data_raw):
        if isinstance(entry, dict) and 'timestamp' in entry and 'buy' in entry:
            try:
                ts = parse_timestamp(entry['timestamp'])
                if most_recent_ts is None or ts > most_recent_ts:
                    most_recent_ts = ts
                    most_recent_entry = entry
            except:
                continue
    
    if most_recent_entry is None:
        raise ValueError("No valid data in provided input.")
    
    most_recent_price = float(most_recent_entry.get('buy', 0))
    
    # Prepare dataframe (computes features from provided data)
    df = prepare_dataframe_from_raw(data_raw, mayor_data)
    if df.empty:
        raise ValueError("No valid data to predict on.")
    
    # Now process for prediction
    df = label_direction(df)
    df.dropna(inplace=True)
    
    if df.empty:
        raise ValueError("No valid data after preprocessing.")
    
    
    # Ensure all feature_columns exist (fill missing with 0)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0
    
    # Keep only feature columns and maintain correct order
    X = df[feature_columns].values
    X = clean_infinite_values(X)
    X_scaled = scaler.transform(X)
    
    # Predict probabilities
    probs = global_model.predict(X_scaled)
    # Use 0.5 threshold for binary classification
    preds = (probs > 0.5).astype(int)
    
    # Get the latest prediction (most recent timestamp)
    latest_idx = -1
    latest_prob = float(probs[latest_idx])
    latest_pred = int(preds[latest_idx])
    # Use the actual most recent price from raw data
    current_price = most_recent_price
    
    # Calculate expected price change based on probability
    # If prob > 0.5, we expect price increase
    direction = "UP" if latest_pred == 1 else "DOWN"
    confidence = latest_prob if latest_pred == 1 else (1 - latest_prob)
    
    # Estimate price movement (conservative estimate)
    # Using 0.2% as base threshold from label_direction
    expected_change_pct = confidence * 0.005  # Scale up to ~0.5% max
    if latest_pred == 0:
        expected_change_pct = -expected_change_pct
    
    predicted_price = current_price * (1 + expected_change_pct)
    
    result = {
        'item_id': item_id,
        'current_price': current_price,
        'predicted_price': predicted_price,
        'predicted_change_pct': expected_change_pct * 100,
        'direction': direction,
        'confidence': confidence * 100,
        'raw_probability': latest_prob,
        'timestamp': str(df.iloc[latest_idx]['timestamp']),
        'recommendation': 'BUY' if direction == 'UP' and confidence > 0.6 else ('SELL' if direction == 'DOWN' and confidence > 0.6 else 'HOLD')
    }
    
    return result



# ------------------- Investment Analysis Functions -------------------
def analyze_best_flips(predictions_list, top_n=10):
    """
    Find best flip opportunities: largest spread where SELL ORDER > BUY ORDER.
    Remember: API returns insta-buy/insta-sell, so we flip the interpretation:
    - buy_current = SELL ORDER price (what you'd get placing a sell order)
    - sell_current = BUY ORDER price (what you'd pay placing a buy order)
    
    For profitable flips: sell_current > buy_current (reversed!)
    
    Args:
        predictions_list: List of prediction dictionaries (flattened or nested format)
        top_n: Number of top flips to return
    
    Returns:
        List of flip opportunities sorted by spread potential
    """
    flip_opportunities = []
    
    for pred in predictions_list:
        try:
            # Support both nested and flattened formats
            if 'buy' in pred and isinstance(pred['buy'], dict):
                # Nested format from predict_item_three_models
                sell_order_price = pred['buy']['current_price']  # What you GET when selling
                buy_order_price = pred['sell']['current_price']   # What you PAY when buying
                buy_order_predicted = pred['sell']['predicted_price']
                sell_order_predicted = pred['buy']['predicted_price']
                buy_direction = pred['sell']['direction']
                sell_direction = pred['buy']['direction']
                spread_direction = pred['spread']['direction']
            else:
                # Flattened format from cached predictions
                sell_order_price = pred['buy_current']  # What you GET when selling
                buy_order_price = pred['sell_current']   # What you PAY when buying
                buy_order_predicted = pred['sell_predicted']
                sell_order_predicted = pred['buy_predicted']
                buy_direction = pred['sell_direction']
                sell_direction = pred['buy_direction']
                spread_direction = pred['spread_direction']
            
            # Skip if buy_order_price is zero or negative (invalid data)
            if buy_order_price <= 0:
                continue
            
            # Only consider if profitable (sell order > buy order)
            if sell_order_price > buy_order_price:
                spread = sell_order_price - buy_order_price
                spread_pct = (spread / buy_order_price) * 100
                
                flip_opportunities.append({
                    'item_id': pred['item_id'],
                    'buy_order_price': buy_order_price,
                    'sell_order_price': sell_order_price,
                    'spread': spread,
                    'spread_pct': spread_pct,
                    'buy_order_predicted': buy_order_predicted,
                    'sell_order_predicted': sell_order_predicted,
                    'buy_direction': buy_direction,
                    'sell_direction': sell_direction,
                    'spread_direction': spread_direction
                })
        except (KeyError, TypeError, ZeroDivisionError) as e:
            # Skip items with missing or invalid data
            print(f"Skipping item {pred.get('item_id', 'unknown')}: {e}")
            continue
    
    # Sort by spread percentage (highest first)
    flip_opportunities.sort(key=lambda x: x['spread_pct'], reverse=True)
    
    return flip_opportunities[:top_n]


def analyze_best_investments(predictions_list, timeframe_days=1, top_n=10):
    """
    Find best investment opportunities with weighted expected return.
    We want SELL ORDER price increases (which is buy_current in API terms).
    
    Weighted Return = probability_of_increase * expected_increase_pct
    
    Args:
        predictions_list: List of prediction dictionaries (flattened or nested format)
        timeframe_days: Investment timeframe (1, 7, or 30 days)
        top_n: Number of top investments to return
    
    Returns:
        List of investment opportunities sorted by weighted expected return
    """
    investments = []
    
    # Scale factor based on timeframe (rough approximation)
    timeframe_multiplier = {
        1: 1.0,    # 1 day baseline
        7: 2.5,    # 1 week: ~2.5x the daily movement
        30: 6.0    # 1 month: ~6x the daily movement (diminishing returns)
    }.get(timeframe_days, 1.0)
    
    for pred in predictions_list:
        # Support both nested and flattened formats
        if 'buy' in pred and isinstance(pred['buy'], dict):
            # Nested format from predict_item_three_models
            # BUY block now represents local TOP signals on sell-order price.
            sell_order_current = pred['buy']['current_price']
            sell_order_predicted = pred['buy']['predicted_price']
            sell_order_change_pct = pred['buy']['change_pct']
            sell_order_confidence = pred['buy']['confidence'] / 100.0
            sell_order_direction = pred['buy']['direction']  # "TOP" or "NONE"
        else:
            # Flattened format
            sell_order_current = pred['buy_current']
            sell_order_predicted = pred['buy_predicted']
            sell_order_change_pct = pred['buy_change_pct']
            sell_order_confidence = pred['buy_confidence'] / 100.0
            sell_order_direction = pred['buy_direction']
        
        # Only consider local TOP signals as potential exit points
        if sell_order_direction == 'TOP':
            # Apply timeframe multiplier to expected change (negative change_pct means expected drop)
            scaled_change_pct = sell_order_change_pct * timeframe_multiplier
            
            # For ranking, we still use magnitude of expected move weighted by confidence
            weighted_return = sell_order_confidence * abs(scaled_change_pct)
            
            investments.append({
                'item_id': pred['item_id'],
                'current_price': sell_order_current,
                'predicted_price': sell_order_current * (1 + scaled_change_pct / 100),
                'expected_change_pct': scaled_change_pct,
                'confidence': sell_order_confidence * 100,
                'weighted_return': weighted_return,
                'timeframe_days': timeframe_days
            })
    
    # Sort by weighted return (highest first)
    investments.sort(key=lambda x: x['weighted_return'], reverse=True)
    
    return investments[:top_n]


def analyze_crash_watch(predictions_list, top_n=10):
    """
    Find items with predicted BUY ORDER price crashes and estimate reversal.
    BUY ORDER = sell_current in API terms (what you'd pay to buy via order).
    
    We look for strong DOWN predictions and estimate when they'll reverse.
    
    Args:
        predictions_list: List of prediction dictionaries (flattened or nested format)
        top_n: Number of crash items to track
    
    Returns:
        List of crashing items with reversal estimates
    """
    crash_items = []
    
    for pred in predictions_list:
        try:
            # Support both nested and flattened formats
            if 'sell' in pred and isinstance(pred['sell'], dict):
                # Nested format from predict_item_three_models
                # SELL block now represents local BOTTOM signals on buy-order price.
                buy_order_current = pred['sell']['current_price']
                buy_order_predicted = pred['sell']['predicted_price']
                buy_order_change_pct = pred['sell']['change_pct']
                buy_order_confidence = pred['sell']['confidence']
                buy_order_direction = pred['sell']['direction']  # "BOTTOM" or "NONE"
                spread_direction = pred['spread']['direction']
                spread_confidence = pred['spread']['confidence']
            else:
                # Flattened format
                buy_order_current = pred['sell_current']
                buy_order_predicted = pred['sell_predicted']
                buy_order_change_pct = pred['sell_change_pct']
                buy_order_confidence = pred['sell_confidence']
                buy_order_direction = pred['sell_direction']
                spread_direction = pred['spread_direction']
                spread_confidence = pred['spread_confidence']
            
            # Only consider BOTTOM signals (local minima) with high confidence
            if buy_order_direction == 'BOTTOM' and buy_order_confidence > 50.0:
                # Estimate crash severity
                crash_severity = abs(buy_order_change_pct) * (buy_order_confidence / 100.0)
                
                # Estimate reversal timing (rough heuristic)
                # Strong crash (high confidence) = faster reversal
                # Estimate in hours: inverse of confidence (higher confidence = sooner reversal expected)
                estimated_reversal_hours = int(24 / max(buy_order_confidence / 100.0, 0.5))
                
                crash_items.append({
                    'item_id': pred['item_id'],
                    'current_price': buy_order_current,
                    'predicted_price': buy_order_predicted,
                    'crash_pct': buy_order_change_pct,
                    'confidence': buy_order_confidence,
                    'crash_severity': crash_severity,
                    'estimated_reversal_hours': estimated_reversal_hours,
                    'spread_direction': spread_direction,
                    'spread_confidence': spread_confidence,
                    'recommendation': 'WAIT' if estimated_reversal_hours > 12 else 'BUY_DIP'
                })
        
        except (KeyError, TypeError, ZeroDivisionError) as e:
            # Skip items with missing or invalid data
            continue
    
    # Sort by crash severity (most severe first)
    crash_items.sort(key=lambda x: x['crash_severity'], reverse=True)
    
    return crash_items[:top_n]

# ------------------- Example Usage -------------------

if __name__ == '__main__':
    number = 200
    print("\n" + "="*70)
    print(f"OPTIMIZED TRAINING: {number} ITEMS FOR 90%+ F1 SCORE")
    print("="*70)
    
    # Fetch all item IDs
    with open('sorted_by_demand_items.json', 'r') as f:
        item_ids = json.load(f)
    item_ids = item_ids[:number]
    for entry in item_ids:
        print(f"Training model for {entry['item_id']}")
        train_three_model_system(entry['item_id'], horizon_bars=3, threshold=0.001)
