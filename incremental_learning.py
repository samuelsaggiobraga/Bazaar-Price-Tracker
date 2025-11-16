import os
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from LGBMfulldata import (
    prepare_dataframe_from_raw, 
    label_direction, 
    clean_infinite_values
)
from data_utils import load_or_fetch_item_data
from mayor_utils import get_mayor_perks


class IncrementalLearner:
    """
    Handles incremental learning for the LightGBM model.
    Allows updating the model with new data without full retraining.
    """
    
    def __init__(self, model_path='global_lgbm_model.pkl', 
                 scaler_path='global_scaler.pkl',
                 feature_columns_path='global_feature_columns.pkl',
                 item_encoder_path='item_encoder.pkl'):
        """Initialize the incremental learner by loading existing model artifacts."""
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_columns_path = feature_columns_path
        self.item_encoder_path = item_encoder_path
        
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.item_encoder = None
        
        self.update_history = []
        self.last_update_time = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessing artifacts."""
        if not all([os.path.exists(p) for p in [self.model_path, self.scaler_path, 
                                                  self.feature_columns_path, self.item_encoder_path]]):
            raise FileNotFoundError("Model artifacts not found. Train a global model first.")
        
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.feature_columns = joblib.load(self.feature_columns_path)
        self.item_encoder = joblib.load(self.item_encoder_path)
        
        print("Model artifacts loaded successfully.")
    
    def save_model(self):
        """Save the updated model and artifacts."""
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.feature_columns, self.feature_columns_path)
        joblib.dump(self.item_encoder, self.item_encoder_path)
        print("Model artifacts saved successfully.")
    
    def update_with_new_data(self, item_id, new_data_raw, horizon_bars=1, 
                           threshold=0.002, num_boost_round=50):
        """
        Incrementally update the model with new data for a specific item.
        
        Args:
            item_id: The item ID to update
            new_data_raw: Raw data entries from the API
            horizon_bars: Prediction horizon
            threshold: Price change threshold for labeling
            num_boost_round: Number of additional boosting rounds
        
        Returns:
            dict: Update statistics
        """
        print(f"Updating model with new data for {item_id}...")
        
        # Prepare the new data
        mayor_data = get_mayor_perks()
        df = prepare_dataframe_from_raw(new_data_raw, mayor_data)
        
        if df.empty:
            return {'status': 'error', 'message': 'No valid data to update with'}
        
        df = label_direction(df, horizon_bars, horizon_bars * threshold)
        df.dropna(inplace=True)
        
        if len(df) < 10:
            return {'status': 'error', 'message': f'Insufficient data points: {len(df)}'}
        
        # Add item_id feature
        df['item_id_int'] = self.item_encoder.transform([item_id] * len(df))
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        X = df[self.feature_columns].values
        y = df['target'].values
        
        # Clean and scale
        X = clean_infinite_values(X)
        X_scaled = self.scaler.transform(X)
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X_scaled, label=y)
        
        # Incremental training (continue from existing model)
        self.model = lgb.train(
            params={
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'learning_rate': 0.01  # Lower learning rate for incremental updates
            },
            train_set=train_data,
            num_boost_round=num_boost_round,
            init_model=self.model
        )
        
        # Track update
        update_info = {
            'timestamp': datetime.now().isoformat(),
            'item_id': item_id,
            'num_samples': len(df),
            'num_boost_rounds': num_boost_round
        }
        self.update_history.append(update_info)
        self.last_update_time = datetime.now()
        
        # Save updated model
        self.save_model()
        
        return {
            'status': 'success',
            'message': f'Model updated with {len(df)} samples',
            'update_info': update_info
        }
    
    def update_multiple_items(self, item_ids, num_boost_round=50):
        """
        Update the model with recent data from multiple items.
        
        Args:
            item_ids: List of item IDs to update
            num_boost_round: Number of boosting rounds per item
        
        Returns:
            list: Update results for each item
        """
        results = []
        
        for item_id in item_ids:
            try:
                # Fetch latest data
                data = load_or_fetch_item_data(item_id, fetch_if_missing=False)
                
                if data is None:
                    results.append({
                        'item_id': item_id,
                        'status': 'error',
                        'message': 'No data available'
                    })
                    continue
                
                # Use only recent data (last 30 days)
                cutoff = datetime.now() - timedelta(days=30)
                recent_data = [
                    entry for entry in data 
                    if datetime.fromisoformat(entry.get('timestamp', '').replace('Z', '')) > cutoff
                ]
                
                if len(recent_data) < 10:
                    results.append({
                        'item_id': item_id,
                        'status': 'error',
                        'message': f'Insufficient recent data: {len(recent_data)}'
                    })
                    continue
                
                # Update model
                result = self.update_with_new_data(
                    item_id, 
                    recent_data, 
                    num_boost_round=num_boost_round
                )
                result['item_id'] = item_id
                results.append(result)
                
            except Exception as e:
                results.append({
                    'item_id': item_id,
                    'status': 'error',
                    'message': str(e)
                })
        
        return results
    
    def get_update_history(self):
        """Return the update history."""
        return {
            'total_updates': len(self.update_history),
            'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
            'history': self.update_history[-10:]  # Last 10 updates
        }


if __name__ == '__main__':
    # Example usage
    learner = IncrementalLearner()
    
    # Test incremental update with a single item
    import requests
    url = "https://sky.coflnet.com/api/items/bazaar/tags"
    item_ids = requests.get(url).json()
    
    test_item = item_ids[0]
    data = load_or_fetch_item_data(test_item, fetch_if_missing=False)
    
    if data:
        result = learner.update_with_new_data(test_item, data[-500:])  # Use last 500 entries
        print("\nUpdate Result:")
        print(json.dumps(result, indent=2))
        
        print("\nUpdate History:")
        print(json.dumps(learner.get_update_history(), indent=2))
