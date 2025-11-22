import optuna
from optuna.integration import LightGBMPruningCallback
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler, LabelEncoder
import gc
import json
from LGBMfulldata import (
    prepare_dataframe_from_raw,
    label_direction,
    label_spread_direction,
    clean_infinite_values,
    optimize_threshold,
    get_mayor_perks
)
from tqdm import tqdm


def load_item_features(item_id, mayor_data, model_type='buy'):
    """Load pre-saved .pkl.gz data for an item and prepare features."""
    try:
        df_base = pd.read_pickle(f"/Users/samuelbraga/Json Files/bazaar_history_{item_id}.pkl.gz")
        df_base = prepare_dataframe_from_raw(df_base, mayor_data, has_mayor_system=True)
        if df_base.empty or len(df_base) < 50:
            return None
        
        if model_type == 'spread':
            df = label_spread_direction(df_base, horizon_bars=1, threshold=0.005)
            target_col = 'target_spread'
        else:
            df = label_direction(df_base, horizon_bars=1, threshold=0.005, price_type=model_type)
            target_col = f'target_{model_type}'
        
        df.dropna(inplace=True)
        if len(df) < 50:
            return None
        
        exclude_cols = set([c for c in df.columns if c.startswith('future_') or c.startswith('target_')]) | {'timestamp'}
        features = [c for c in df.columns if c not in exclude_cols]
        
        X = df[features].values
        y = df[target_col].values
        X = clean_infinite_values(X)
        
        # Temporal split
        split_idx = int(len(X) * 0.85)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        del df_base, df, X, y
        gc.collect()
        
        return X_train, y_train, X_val, y_val, features
    except Exception as e:
        print(f"Error loading item {item_id}: {e}")
        return None


def objective(trial, item_ids, mayor_data, model_type='buy'):
    """Run the trial objective per item, deleting memory immediately."""
    
    all_f1 = []
    features = None
    
    for item_id in tqdm(item_ids, desc=f"Trial {trial.number}"):
        item_data = load_item_features(item_id, mayor_data, model_type)
        if item_data is None:
            continue
        
        X_train, y_train, X_val, y_val, features = item_data
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Class weights
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()
        scale_pos_weight = neg_count / (pos_count + 1e-9)
        
        # Hyperparameters
        param = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.1),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),
            'scale_pos_weight': scale_pos_weight,
            'seed': 42,
            'force_col_wise': True
        }
        
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
        pruning_callback = LightGBMPruningCallback(trial, 'auc')
        
        model = lgb.train(
            param,
            train_data,
            valid_sets=[val_data],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(stopping_rounds=50), pruning_callback, lgb.log_evaluation(period=0)]
        )
        
        y_pred_proba = model.predict(X_val_scaled)
        _, best_f1 = optimize_threshold(y_val, y_pred_proba)
        all_f1.append(best_f1)
        
        # Clean up memory immediately
        del X_train, y_train, X_val, y_val, X_train_scaled, X_val_scaled, train_data, val_data, model
        gc.collect()
    
    if not all_f1:
        return 0.0
    return np.mean(all_f1)


def run_optimization(model_type='buy', n_trials=50, n_items=100):
    print(f"Optimizing {model_type} model with {n_trials} trials on {n_items} items...")
    
    # Load item list
    url = "https://sky.coflnet.com/api/items/bazaar/tags"
    item_ids = requests.get(url).json()[:n_items]
    mayor_data = get_mayor_perks()
    
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(lambda trial: objective(trial, item_ids, mayor_data, model_type), n_trials=n_trials)
    
    print(f"Best F1: {study.best_value:.4f}")
    print("Best hyperparameters:")
    print(study.best_params)
    
    with open(f'best_params_{model_type}.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    return study.best_params


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Optimize LightGBM hyperparameters')
    parser.add_argument('--model', type=str, default='buy', choices=['buy','sell','spread'])
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--n-items', type=int, default=1417)
    args = parser.parse_args()

    best_params = run_optimization(
        model_type=args.model,
        n_trials=args.n_trials,
        n_items=args.n_items
    )
