import json
import os
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from train_model import (
    get_mayor_perks, match_mayor_perks, 
    load_item_model, save_model, train_single_item_model
)


class IncrementalTrainer:
    """
    Manages incremental retraining of RandomForest models.
    Uses sliding window approach: retrain on recent data + new data.
    """
    
    def __init__(self, item_id, model_dir="models", data_dir="."):
        self.item_id = item_id
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model = None
        self.scaler = None
        self.metadata = None
        self.mayor_data = None
        
        os.makedirs(model_dir, exist_ok=True)
        
    def load_existing_model(self):
        """Load existing trained model for this item."""
        self.model, self.scaler, self.metadata = load_item_model(
            self.item_id, self.model_dir
        )
        
        if self.model is None:
            print(f"No existing model for {self.item_id}")
            return False
        
        print(f"Loaded model for {self.item_id}")
        print(f"  Trained at: {self.metadata.get('trained_at', 'unknown')}")
        print(f"  Total samples: {self.metadata.get('total_samples', 'unknown')}")
        return True
    
    def prepare_features(self, entry, mayor_data):
        """Extract features from bazaar data entry."""
        timestamp = entry.get('timestamp')
        if not timestamp:
            return None, None, None
        
        try:
            # Price and volume features
            buy_price = float(entry.get('buy', 0))
            sell_price = float(entry.get('sell', 0))
            buy_volume = float(entry.get('buyVolume', 0))
            sell_volume = float(entry.get('sellVolume', 0))
            buy_moving_week = float(entry.get('buyMovingWeek', 0))
            sell_moving_week = float(entry.get('sellMovingWeek', 0))
            max_buy = float(entry.get('maxBuy', 0))
            max_sell = float(entry.get('maxSell', 0))
            min_buy = float(entry.get('minBuy', 0))
            min_sell = float(entry.get('minSell', 0))
            
            # Date features
            year = int(timestamp[:4])
            month = int(timestamp[5:7])
            day = int(timestamp[8:10])
            
            date_obj = datetime.strptime(timestamp[:10], '%Y-%m-%d')
            day_of_year = date_obj.timetuple().tm_yday
            day_of_week = date_obj.weekday()
            
            # Mayor perks
            mayor_perks = match_mayor_perks(timestamp, mayor_data)
            if mayor_perks is None:
                mayor_perks = [0] * 40
            
            # Combine features
            feature_vector = [
                year, month, day, day_of_year, day_of_week,
                buy_price, sell_price,
                buy_volume, sell_volume,
                buy_moving_week, sell_moving_week,
                max_buy, max_sell,
                min_buy, min_sell
            ] + mayor_perks
            
            return feature_vector, buy_price, timestamp
            
        except (ValueError, TypeError):
            return None, None, None
    
    def load_historical_data(self, days_back=30):
        """Load recent historical data from JSON file."""
        filename = os.path.join(self.data_dir, f"bazaar_history_combined_{self.item_id}.json")
        
        if not os.path.exists(filename):
            print(f"No historical data file found: {filename}")
            return []
        
        with open(filename, 'r') as f:
            all_data = json.load(f)
        
        # Filter to recent data
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_data = []
        
        for entry in all_data:
            if not isinstance(entry, dict):
                continue
            timestamp = entry.get('timestamp')
            if not timestamp:
                continue
            
            try:
                entry_date = datetime.strptime(timestamp[:10], '%Y-%m-%d')
                if entry_date >= cutoff_date:
                    recent_data.append(entry)
            except:
                continue
        
        print(f"Loaded {len(recent_data)} recent data points (last {days_back} days)")
        return recent_data
    
    def append_new_data(self, new_entries):
        """Append new data entries to the historical JSON file."""
        filename = os.path.join(self.data_dir, f"bazaar_history_combined_{self.item_id}.json")
        
        # Load existing data
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                all_data = json.load(f)
        else:
            all_data = []
        
        # Append new entries
        all_data.extend(new_entries)
        
        # Save back
        with open(filename, 'w') as f:
            json.dump(all_data, f)
        
        print(f"Appended {len(new_entries)} new entries to {filename}")
    
    def retrain_with_new_data(self, new_data, sliding_window_days=30):
        """
        Retrain the model with recent historical + new data.
        
        Args:
            new_data: List of new bazaar data entries
            sliding_window_days: Number of days of history to include
            
        Returns:
            True if retraining successful, False otherwise
        """
        print(f"\n=== Retraining {self.item_id} ===")
        
        # Fetch mayor data if needed
        if self.mayor_data is None:
            self.mayor_data = get_mayor_perks()
        
        # Load recent historical data
        historical_data = self.load_historical_data(days_back=sliding_window_days)
        
        # Combine with new data
        combined_data = historical_data + new_data
        
        if len(combined_data) < 10:
            print(f"Not enough data to retrain ({len(combined_data)} samples)")
            return False
        
        # Prepare features
        features_list = []
        targets_list = []
        timestamps_list = []
        
        for entry in combined_data:
            feature_vec, target, timestamp = self.prepare_features(entry, self.mayor_data)
            if feature_vec is not None:
                features_list.append(feature_vec)
                targets_list.append(target)
                timestamps_list.append(timestamp)
        
        if len(features_list) == 0:
            print("No valid features extracted")
            return False
        
        X = np.array(features_list)
        y = np.array(targets_list)
        
        print(f"Training on {len(X)} samples")
        
        # Create fresh scaler and retrain model
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        
        # Update metadata
        self.metadata = {
            'item_id': self.item_id,
            'total_samples': len(X),
            'trained_at': datetime.now().isoformat(),
            'feature_count': X.shape[1],
            'sliding_window_days': sliding_window_days
        }
        
        # Save updated model
        model_path = os.path.join(self.model_dir, self.item_id)
        save_model(self.model, self.scaler, model_path, self.metadata)
        
        print(f"Retraining complete! Model saved.")
        return True
    
    def predict(self, entry):
        """Make prediction for a bazaar entry."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        if self.mayor_data is None:
            self.mayor_data = get_mayor_perks()
        
        feature_vec, _, _ = self.prepare_features(entry, self.mayor_data)
        if feature_vec is None:
            return None
        
        feature_vec = np.array(feature_vec).reshape(1, -1)
        feature_scaled = self.scaler.transform(feature_vec)
        
        return self.model.predict(feature_scaled)[0]


def retrain_item(item_id, new_data, model_dir="models", data_dir=".", sliding_window_days=30):
    """
    Convenience function to retrain a single item model.
    
    Args:
        item_id: Bazaar item ID
        new_data: List of new data entries
        model_dir: Directory containing models
        data_dir: Directory containing historical data files
        sliding_window_days: Days of history to include
        
    Returns:
        IncrementalTrainer instance
    """
    trainer = IncrementalTrainer(item_id, model_dir, data_dir)
    
    # Try to load existing model
    trainer.load_existing_model()
    
    # Append new data to historical file
    if len(new_data) > 0:
        trainer.append_new_data(new_data)
    
    # Retrain
    success = trainer.retrain_with_new_data(new_data, sliding_window_days)
    
    return trainer if success else None
