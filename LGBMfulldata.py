import os
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
from data_utils import load_or_fetch_item_data
from mayor_utils import get_mayor_perks, match_mayor_perks, get_mayor_start_date
import requests
import gc

# ------------------- Helpers -------------------

def parse_timestamp(ts_str):
    fmts = ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")
    for fmt in fmts:
        try:
            return datetime.strptime(ts_str, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(ts_str)
    except Exception:
        raise ValueError(f"Unrecognized timestamp format: {ts_str}")

def add_time_features(df, ts_col='timestamp'):
    dt = pd.to_datetime(df[ts_col])
    df['hour'] = dt.dt.hour
    df['minute'] = dt.dt.minute
    # cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute']/60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute']/60)
    df['dayofweek'] = dt.dt.dayofweek
    df['dayofweek_sin'] = np.sin(2*np.pi*df['dayofweek']/7)
    df['dayofweek_cos'] = np.cos(2*np.pi*df['dayofweek']/7)
    return df

def build_lagged_features(df, price_col='buy_price', vol_col='buy_volume', lags=(1,2,3,6,12)):
    df['ret'] = df[price_col].pct_change()
    for lag in lags:
        df[f'ret_lag_{lag}'] = df['ret'].shift(lag)
        df[f'price_lag_{lag}'] = df[price_col].shift(lag)
        df[f'vol_lag_{lag}'] = df[vol_col].shift(lag)
    windows = [3,6,12]
    for w in windows:
        df[f'roll_mean_{w}'] = df['ret'].rolling(w).mean()
        df[f'roll_std_{w}'] = df['ret'].rolling(w).std()
        if w>=6:
            df[f'roll_skew_{w}'] = df['ret'].rolling(w).skew()
            df[f'roll_kurt_{w}'] = df['ret'].rolling(w).kurt()
    df['mom_3'] = df['ret'].rolling(3).sum()
    df['mom_6'] = df['ret'].rolling(6).sum()
    df['price_rolling_mean_12'] = df[price_col].rolling(12).mean()
    df['price_zscore_12'] = (df[price_col]-df['price_rolling_mean_12'])/(df['ret'].rolling(12).std()+1e-9)
    return df

def prepare_dataframe_from_raw(data, mayor_data=None):
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
            'min_sell': fget('minSell')
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
    df = build_lagged_features(df)
    return df

def label_direction(df, horizon_bars=1, threshold=0.002):
    df=df.copy()
    df[f'future_price_{horizon_bars}'] = df['buy_price'].shift(-horizon_bars)
    df[f'future_ret_{horizon_bars}'] = (df[f'future_price_{horizon_bars}'] - df['buy_price']) / df['buy_price']
    df['target'] = (df[f'future_ret_{horizon_bars}'] > threshold).astype(int)
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

# ------------------- Global Model Training -------------------

def train_global_model(item_ids, horizon_bars=1, threshold=0.002, use_mayor_data=True, 
                      n_folds=5, train_val_split=0.8):
    """
    Train global LightGBM model with cross-validation.
    
    Args:
        item_ids: List of item IDs to train on
        horizon_bars: Prediction horizon
        threshold: Price change threshold for labeling
        use_mayor_data: Whether mayor data is available (affects train/val split)
        n_folds: Number of folds for cross-validation
        train_val_split: Train/validation split ratio (0.8 for 80/20, 0.7 for 70/30)
    """
    split_name = f"{int(train_val_split*100)}/{int((1-train_val_split)*100)}"
    print(f"Training global LightGBM model with {n_folds}-fold CV ({split_name} split)...")
    print(f"Mayor data: {'Available' if use_mayor_data else 'Not Available'}")
    
    mayor_data = get_mayor_perks() if use_mayor_data else None
    item_encoder = LabelEncoder()
    item_encoder.fit(item_ids)
    
    global_model = None
    feature_columns = None
    scaler = StandardScaler()
    
    # Collect all data first for proper CV
    all_X = []
    all_y = []
    
    for idx, item_id in enumerate(item_ids):
        print(f"\nProcessing item {idx+1}/{len(item_ids)}: {item_id}")
        data = load_or_fetch_item_data(item_id)
        if data is None:
            continue
        df = prepare_dataframe_from_raw(data, mayor_data)
        if df.empty:
            continue
        df = label_direction(df, horizon_bars, horizon_bars*threshold)
        df.dropna(inplace=True)
        if len(df) < 50:
            continue
        
        # add item_id as a feature
        df['item_id_int'] = item_encoder.transform([item_id]*len(df))
        
        exclude_cols = set([c for c in df.columns if c.startswith('future_price') or c.startswith('future_ret')]) | {'timestamp','target'}
        curr_features = [c for c in df.columns if c not in exclude_cols]
        if feature_columns is None:
            feature_columns = curr_features
        else:
            curr_features = feature_columns
        
        X = df[curr_features].values
        y = df['target'].values
        
        # Clean infinite and too large values
        X = clean_infinite_values(X)
        
        all_X.append(X)
        all_y.append(y)
        
        # free memory
        del df, X, y, data
        gc.collect()
    
    if not all_X:
        print("No valid data found!")
        return None, None, None, None
    
    # Concatenate all data
    X_combined = np.vstack(all_X)
    y_combined = np.hstack(all_y)
    
    print(f"\nTotal samples: {len(X_combined)}")
    print(f"Starting {n_folds}-fold cross-validation...\n")
    
    # Perform stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_metrics = []
    best_model = None
    best_auc = 0
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_combined), 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}/{n_folds}")
        print(f"{'='*60}")
        
        # Split based on train_val_split ratio
        train_size = int(len(train_idx) * train_val_split)
        actual_train_idx = train_idx[:train_size]
        actual_val_idx = np.concatenate([train_idx[train_size:], val_idx])
        
        X_train, X_val = X_combined[actual_train_idx], X_combined[actual_val_idx]
        y_train, y_val = y_combined[actual_train_idx], y_combined[actual_val_idx]
        
        print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Fit scaler on training data only
        fold_scaler = StandardScaler()
        X_train_scaled = fold_scaler.fit_transform(X_train)
        X_val_scaled = fold_scaler.transform(X_val)
        
        # Train model
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
        
        fold_model = lgb.train(
            params={
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'feature_fraction': 0.8
            },
            train_set=train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            valid_names=['validation']
        )
        
        # Predictions
        y_pred_proba = fold_model.predict(X_val_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        fold_metrics.append({
            'fold': fold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        })
        
        print(f"\nFold {fold} Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
        
        # Keep best model based on AUC
        if auc > best_auc:
            best_auc = auc
            best_model = fold_model
            scaler = fold_scaler
        
        del train_data, val_data, X_train_scaled, X_val_scaled
        gc.collect()
    
    # Print average metrics
    print(f"\n{'='*60}")
    print("Cross-Validation Summary")
    print(f"{'='*60}")
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
    avg_precision = np.mean([m['precision'] for m in fold_metrics])
    avg_recall = np.mean([m['recall'] for m in fold_metrics])
    avg_f1 = np.mean([m['f1'] for m in fold_metrics])
    avg_auc = np.mean([m['auc'] for m in fold_metrics])
    
    std_accuracy = np.std([m['accuracy'] for m in fold_metrics])
    std_precision = np.std([m['precision'] for m in fold_metrics])
    std_recall = np.std([m['recall'] for m in fold_metrics])
    std_f1 = np.std([m['f1'] for m in fold_metrics])
    std_auc = np.std([m['auc'] for m in fold_metrics])
    
    print(f"\nAverage Metrics ({n_folds}-fold):")
    print(f"  Accuracy:  {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"  Precision: {avg_precision:.4f} ± {std_precision:.4f}")
    print(f"  Recall:    {avg_recall:.4f} ± {std_recall:.4f}")
    print(f"  F1-Score:  {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"  AUC:       {avg_auc:.4f} ± {std_auc:.4f}")
    
    # Train final model on all data
    print(f"\n{'='*60}")
    print("Training final model on all data...")
    print(f"{'='*60}")
    
    X_scaled = scaler.fit_transform(X_combined)
    train_data = lgb.Dataset(X_scaled, label=y_combined)
    
    global_model = lgb.train(
        params={
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8
        },
        train_set=train_data,
        num_boost_round=200
    )
    
    # save global model
    joblib.dump(global_model, 'global_lgbm_model.pkl')
    joblib.dump(scaler, 'global_scaler.pkl')
    joblib.dump(feature_columns, 'global_feature_columns.pkl')
    joblib.dump(item_encoder, 'item_encoder.pkl')
    
    # Save metrics
    metrics_summary = {
        'n_folds': n_folds,
        'train_val_split': f"{split_name}",
        'mayor_data_used': use_mayor_data,
        'fold_metrics': fold_metrics,
        'average_metrics': {
            'accuracy': float(avg_accuracy),
            'precision': float(avg_precision),
            'recall': float(avg_recall),
            'f1': float(avg_f1),
            'auc': float(avg_auc)
        },
        'std_metrics': {
            'accuracy': float(std_accuracy),
            'precision': float(std_precision),
            'recall': float(std_recall),
            'f1': float(std_f1),
            'auc': float(std_auc)
        }
    }
    
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print("\nGlobal model training complete!")
    print("Metrics saved to model_metrics.json")
    
    return global_model, scaler, feature_columns, item_encoder


def train_two_phase_model(item_ids, horizon_bars=1, threshold=0.002):
    """
    Two-phase training based on mayor data availability:
    Phase 1: Train on data BEFORE mayor data (3-fold CV, 80/20 split, no mayor features)
    Phase 2: Train on data AFTER mayor data (5-fold CV, 70/30 split, with mayor features)
    
    Args:
        item_ids: List of item IDs to train on
        horizon_bars: Prediction horizon
        threshold: Price change threshold for labeling
    """
    print("\n" + "="*70)
    print("TWO-PHASE TEMPORAL TRAINING")
    print("Phase 1: Pre-Mayor Data (3-fold CV, 80/20 split)")
    print("Phase 2: Post-Mayor Data (5-fold CV, 70/30 split)")
    print("="*70 + "\n")
    
    # Get mayor data to determine split point
    mayor_data = get_mayor_perks()
    mayor_start_date = get_mayor_start_date(mayor_data)
    
    if mayor_start_date is None:
        print("WARNING: No mayor data found. Training single-phase model without mayor data.")
        return train_global_model(item_ids, horizon_bars, threshold, 
                                use_mayor_data=False, n_folds=3, train_val_split=0.8)
    
    print(f"Mayor data starts from: {mayor_start_date.strftime('%Y-%m-%d')}")
    print(f"Splitting data at this timestamp...\n")
    
    item_encoder = LabelEncoder()
    item_encoder.fit(item_ids)
    
    # Collect data for both phases
    phase1_X = []  # Pre-mayor data (no mayor features)
    phase1_y = []
    phase2_X = []  # Post-mayor data (with mayor features)
    phase2_y = []
    
    feature_columns_phase1 = None
    feature_columns_phase2 = None
    
    for idx, item_id in enumerate(item_ids):
        print(f"Processing item {idx+1}/{len(item_ids)}: {item_id}")
        data = load_or_fetch_item_data(item_id)
        if data is None:
            continue
        
        # Separate data by mayor availability
        pre_mayor_data = []
        post_mayor_data = []
        
        for entry in data:
            if not isinstance(entry, dict) or 'timestamp' not in entry:
                continue
            try:
                ts_str = entry['timestamp']
                ts_dt = parse_timestamp(ts_str)
                
                if ts_dt < mayor_start_date:
                    pre_mayor_data.append(entry)
                else:
                    post_mayor_data.append(entry)
            except:
                continue
        
        # Process Phase 1 data (pre-mayor)
        if pre_mayor_data:
            df1 = prepare_dataframe_from_raw(pre_mayor_data, mayor_data=None)
            if not df1.empty:
                df1 = label_direction(df1, horizon_bars, horizon_bars*threshold)
                df1.dropna(inplace=True)
                if len(df1) >= 20:
                    df1['item_id_int'] = item_encoder.transform([item_id]*len(df1))
                    
                    exclude_cols = set([c for c in df1.columns if c.startswith('future_price') or c.startswith('future_ret')]) | {'timestamp','target'}
                    curr_features = [c for c in df1.columns if c not in exclude_cols]
                    
                    if feature_columns_phase1 is None:
                        feature_columns_phase1 = curr_features
                    else:
                        curr_features = feature_columns_phase1
                    
                    X1 = df1[curr_features].values
                    y1 = df1['target'].values
                    X1 = clean_infinite_values(X1)
                    
                    phase1_X.append(X1)
                    phase1_y.append(y1)
        
        # Process Phase 2 data (post-mayor)
        if post_mayor_data:
            df2 = prepare_dataframe_from_raw(post_mayor_data, mayor_data)
            if not df2.empty:
                df2 = label_direction(df2, horizon_bars, horizon_bars*threshold)
                df2.dropna(inplace=True)
                if len(df2) >= 20:
                    df2['item_id_int'] = item_encoder.transform([item_id]*len(df2))
                    
                    exclude_cols = set([c for c in df2.columns if c.startswith('future_price') or c.startswith('future_ret')]) | {'timestamp','target'}
                    curr_features = [c for c in df2.columns if c not in exclude_cols]
                    
                    if feature_columns_phase2 is None:
                        feature_columns_phase2 = curr_features
                    else:
                        curr_features = feature_columns_phase2
                    
                    X2 = df2[curr_features].values
                    y2 = df2['target'].values
                    X2 = clean_infinite_values(X2)
                    
                    phase2_X.append(X2)
                    phase2_y.append(y2)
        
        del data
        gc.collect()
    
    # ========== PHASE 1: Pre-Mayor Training ==========
    print("\n" + "="*70)
    print("PHASE 1: Training on Pre-Mayor Data")
    print("Configuration: 3-fold CV, 80/20 split, NO mayor features")
    print("="*70 + "\n")
    
    phase1_model = None
    phase1_scaler = StandardScaler()
    phase1_metrics = None
    
    if phase1_X:
        X_phase1 = np.vstack(phase1_X)
        y_phase1 = np.hstack(phase1_y)
        print(f"Phase 1 samples: {len(X_phase1)}")
        
        # 3-fold CV with 80/20 split
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_phase1, y_phase1), 1):
            print(f"\nFold {fold}/3")
            
            train_size = int(len(train_idx) * 0.8)
            actual_train_idx = train_idx[:train_size]
            actual_val_idx = np.concatenate([train_idx[train_size:], val_idx])
            
            X_train, X_val = X_phase1[actual_train_idx], X_phase1[actual_val_idx]
            y_train, y_val = y_phase1[actual_train_idx], y_phase1[actual_val_idx]
            
            print(f"Train: {len(X_train)}, Val: {len(X_val)}")
            
            fold_scaler = StandardScaler()
            X_train_scaled = fold_scaler.fit_transform(X_train)
            X_val_scaled = fold_scaler.transform(X_val)
            
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
            
            fold_model = lgb.train(
                params={'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
                       'learning_rate': 0.05, 'num_leaves': 31, 'feature_fraction': 0.8},
                train_set=train_data,
                num_boost_round=200,
                valid_sets=[val_data],
                valid_names=['validation']
            )
            
            y_pred_proba = fold_model.predict(X_val_scaled)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            auc = roc_auc_score(y_val, y_pred_proba)
            
            fold_metrics.append({'fold': fold, 'accuracy': accuracy, 'precision': precision,
                               'recall': recall, 'f1': f1, 'auc': auc})
            
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                  f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            if fold == 1:
                phase1_model = fold_model
                phase1_scaler = fold_scaler
            
            del train_data, val_data, X_train_scaled, X_val_scaled
            gc.collect()
        
        # Calculate Phase 1 averages
        phase1_metrics = {
            'n_folds': 3,
            'train_val_split': '80/20',
            'fold_metrics': fold_metrics,
            'average_metrics': {
                'accuracy': float(np.mean([m['accuracy'] for m in fold_metrics])),
                'precision': float(np.mean([m['precision'] for m in fold_metrics])),
                'recall': float(np.mean([m['recall'] for m in fold_metrics])),
                'f1': float(np.mean([m['f1'] for m in fold_metrics])),
                'auc': float(np.mean([m['auc'] for m in fold_metrics]))
            }
        }
        
        print(f"\nPhase 1 Average Metrics:")
        print(f"  Accuracy:  {phase1_metrics['average_metrics']['accuracy']:.4f}")
        print(f"  Precision: {phase1_metrics['average_metrics']['precision']:.4f}")
        print(f"  Recall:    {phase1_metrics['average_metrics']['recall']:.4f}")
        print(f"  F1-Score:  {phase1_metrics['average_metrics']['f1']:.4f}")
        print(f"  AUC:       {phase1_metrics['average_metrics']['auc']:.4f}")
    else:
        print("No pre-mayor data available. Skipping Phase 1.")
    
    # ========== PHASE 2: Post-Mayor Training ==========
    print("\n" + "="*70)
    print("PHASE 2: Training on Post-Mayor Data")
    print("Configuration: 5-fold CV, 70/30 split, WITH mayor features")
    print("="*70 + "\n")
    
    phase2_model = None
    phase2_scaler = StandardScaler()
    phase2_metrics = None
    
    if phase2_X:
        X_phase2 = np.vstack(phase2_X)
        y_phase2 = np.hstack(phase2_y)
        print(f"Phase 2 samples: {len(X_phase2)}")
        
        # 5-fold CV with 70/30 split
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_metrics = []
        best_auc = 0
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_phase2, y_phase2), 1):
            print(f"\nFold {fold}/5")
            
            train_size = int(len(train_idx) * 0.7)
            actual_train_idx = train_idx[:train_size]
            actual_val_idx = np.concatenate([train_idx[train_size:], val_idx])
            
            X_train, X_val = X_phase2[actual_train_idx], X_phase2[actual_val_idx]
            y_train, y_val = y_phase2[actual_train_idx], y_phase2[actual_val_idx]
            
            print(f"Train: {len(X_train)}, Val: {len(X_val)}")
            
            fold_scaler = StandardScaler()
            X_train_scaled = fold_scaler.fit_transform(X_train)
            X_val_scaled = fold_scaler.transform(X_val)
            
            train_data = lgb.Dataset(X_train_scaled, label=y_train)
            val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
            
            # Train fresh model (can't use init_model since features differ from Phase 1)
            fold_model = lgb.train(
                params={'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
                       'learning_rate': 0.05, 'num_leaves': 31, 'feature_fraction': 0.8},
                train_set=train_data,
                num_boost_round=200,
                valid_sets=[val_data],
                valid_names=['validation']
            )
            
            y_pred_proba = fold_model.predict(X_val_scaled)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            auc = roc_auc_score(y_val, y_pred_proba)
            
            fold_metrics.append({'fold': fold, 'accuracy': accuracy, 'precision': precision,
                               'recall': recall, 'f1': f1, 'auc': auc})
            
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                  f"Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                phase2_model = fold_model
                phase2_scaler = fold_scaler
            
            del train_data, val_data, X_train_scaled, X_val_scaled
            gc.collect()
        
        # Calculate Phase 2 averages
        phase2_metrics = {
            'n_folds': 5,
            'train_val_split': '70/30',
            'fold_metrics': fold_metrics,
            'average_metrics': {
                'accuracy': float(np.mean([m['accuracy'] for m in fold_metrics])),
                'precision': float(np.mean([m['precision'] for m in fold_metrics])),
                'recall': float(np.mean([m['recall'] for m in fold_metrics])),
                'f1': float(np.mean([m['f1'] for m in fold_metrics])),
                'auc': float(np.mean([m['auc'] for m in fold_metrics]))
            }
        }
        
        print(f"\nPhase 2 Average Metrics:")
        print(f"  Accuracy:  {phase2_metrics['average_metrics']['accuracy']:.4f}")
        print(f"  Precision: {phase2_metrics['average_metrics']['precision']:.4f}")
        print(f"  Recall:    {phase2_metrics['average_metrics']['recall']:.4f}")
        print(f"  F1-Score:  {phase2_metrics['average_metrics']['f1']:.4f}")
        print(f"  AUC:       {phase2_metrics['average_metrics']['auc']:.4f}")
    else:
        print("No post-mayor data available. Skipping Phase 2.")
    
    # ========== Save Final Model ==========
    # Use Phase 2 model if available, otherwise Phase 1
    final_model = phase2_model if phase2_model else phase1_model
    final_scaler = phase2_scaler if phase2_model else phase1_scaler
    final_features = feature_columns_phase2 if phase2_model else feature_columns_phase1
    
    if final_model is None:
        print("\nERROR: No model trained. Insufficient data.")
        return None, None, None, None
    
    joblib.dump(final_model, 'global_lgbm_model.pkl')
    joblib.dump(final_scaler, 'global_scaler.pkl')
    joblib.dump(final_features, 'global_feature_columns.pkl')
    joblib.dump(item_encoder, 'item_encoder.pkl')
    
    # Save comprehensive metrics
    two_phase_metrics = {
        'training_mode': 'two_phase_temporal',
        'mayor_start_date': mayor_start_date.strftime('%Y-%m-%d'),
        'phase1': phase1_metrics,
        'phase2': phase2_metrics
    }
    
    with open('model_metrics.json', 'w') as f:
        json.dump(two_phase_metrics, f, indent=2)
    
    print("\n" + "="*70)
    print("Two-Phase Training Complete!")
    print("="*70)
    print("\nModel artifacts saved:")
    print("  - global_lgbm_model.pkl")
    print("  - global_scaler.pkl")
    print("  - global_feature_columns.pkl")
    print("  - item_encoder.pkl")
    print("  - model_metrics.json")
    
    return final_model, final_scaler, final_features, item_encoder


# ------------------- Prediction -------------------

def predict_item(global_model, scaler, feature_columns, item_encoder, item_id, data_raw, mayor_data=None):
    """
    Predicts the direction for a single item using the global model.
    Returns a user-friendly prediction dictionary with the latest forecast.
    """
    # Get the actual most recent price directly from raw JSON data
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
        raise ValueError("No valid data in raw input.")
    
    most_recent_price = float(most_recent_entry.get('buy', 0))
    most_recent_timestamp = most_recent_ts
    
    # Prepare dataframe
    df = prepare_dataframe_from_raw(data_raw, mayor_data)
    if df.empty:
        raise ValueError("No valid data to predict on.")
    
    # Now process for prediction
    df = label_direction(df)
    df.dropna(inplace=True)
    
    if df.empty:
        raise ValueError("No valid data after preprocessing.")
    
    # Add item_id feature
    df['item_id_int'] = item_encoder.transform([item_id]*len(df))
    
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

# ------------------- Example Usage -------------------

if __name__ == '__main__':
    import sys
    
    # fetch all item IDs
    url = "https://sky.coflnet.com/api/items/bazaar/tags"
    item_ids = requests.get(url).json()
    """
    # Determine training mode based on command line argument
    mode = sys.argv[1] if len(sys.argv) > 1 else 'no_mayor'
    
    if mode == 'no_mayor':
        # Training WITHOUT mayor data (80/20 split, 5-fold CV)
        print("\n" + "="*60)
        print("MODE: Training WITHOUT Mayor Data")
        print("Configuration: 5-fold CV with 80/20 train/val split")
        print("="*60 + "\n")
        model, scaler, feature_columns, item_encoder = train_global_model(
            item_ids[:3],  # first 3 for testing
            use_mayor_data=False,
            n_folds=5,
            train_val_split=0.8
        )
    
    elif mode == 'with_mayor':
        # Training WITH mayor data (70/30 split, 5-fold CV)
        print("\n" + "="*60)
        print("MODE: Training WITH Mayor Data")
        print("Configuration: 5-fold CV with 70/30 train/val split")
        print("="*60 + "\n")
        model, scaler, feature_columns, item_encoder = train_global_model(
            item_ids[:3],  # first 3 for testing
            use_mayor_data=True,
            n_folds=5,
            train_val_split=0.7
        )
    
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python LGBMfulldata.py [no_mayor|with_mayor]")
        sys.exit(1)
    
    if model is None:
        print("Training failed - no valid data found.")
        sys.exit(1)
    
    # Example: predict on single item
    test_item = item_ids[0]
    data = load_or_fetch_item_data(test_item)
    mayor_data = get_mayor_perks() if mode == 'with_mayor' else None
    prediction = predict_item(model, scaler, feature_columns, item_encoder, test_item, data, mayor_data)
    
    print("\n" + "="*60)
    print(f"Prediction for {test_item}")
    print("="*60)
    print(f"Current Price: ${prediction['current_price']:,.2f}")
    print(f"Predicted Price: ${prediction['predicted_price']:,.2f}")
    print(f"Expected Change: {prediction['predicted_change_pct']:.2f}%")
    print(f"Direction: {prediction['direction']}")
    print(f"Confidence: {prediction['confidence']:.1f}%")
    print(f"Recommendation: {prediction['recommendation']}")
    print("="*60)
    """

    model, scaler, feature_columns, item_encoder = train_two_phase_model(item_ids[:3])
    
    test_item = item_ids[0]
    data = load_or_fetch_item_data(test_item)
    mayor_data = get_mayor_perks() 
    prediction = predict_item(model, scaler, feature_columns, item_encoder, test_item, data, mayor_data)
    
    print("\n" + "="*60)
    print(f"Prediction for {test_item}")
    print("="*60)
    print(f"Current Price: ${prediction['current_price']:,.2f}")
    print(f"Predicted Price: ${prediction['predicted_price']:,.2f}")
    print(f"Expected Change: {prediction['predicted_change_pct']:.2f}%")
    print(f"Direction: {prediction['direction']}")
    print(f"Confidence: {prediction['confidence']:.1f}%")
    print(f"Recommendation: {prediction['recommendation']}")
    print("="*60)
    

