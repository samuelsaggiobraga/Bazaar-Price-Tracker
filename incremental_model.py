import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from mayor_utils import get_mayor_perks, match_mayor_perks


class IncrementalBazaarModel:
    """
    Incremental learning model for Bazaar price prediction.
    Uses SGDRegressor with partial_fit for online updates.
    """
    
    def __init__(self, item_id, model_dir="models"):
        self.item_id = item_id
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.last_update = None
        self.total_samples = 0
        self.mayor_data = None
        
        os.makedirs(model_dir, exist_ok=True)
        
    def _get_model_path(self):
        return os.path.join(self.model_dir, f"{self.item_id}_incremental_model.pkl")
    
    def _get_scaler_path(self):
        return os.path.join(self.model_dir, f"{self.item_id}_incremental_scaler.pkl")
    
    def _get_metadata_path(self):
        return os.path.join(self.model_dir, f"{self.item_id}_metadata.pkl")
    
    def initialize_model(self):
        """Initialize a new SGDRegressor model."""
        self.model = SGDRegressor(
            loss='squared_error',
            penalty='l2',
            alpha=0.0001,
            learning_rate='invscaling',
            eta0=0.01,
            power_t=0.25,
            max_iter=1,
            tol=None,
            warm_start=True,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.last_update = datetime.now()
        self.total_samples = 0
        print(f"Initialized new incremental model for {self.item_id}")
    
    def load_model(self):
        """Load existing model from disk."""
        model_path = self._get_model_path()
        scaler_path = self._get_scaler_path()
        metadata_path = self._get_metadata_path()
        
        if not os.path.exists(model_path):
            print(f"No existing model found for {self.item_id}, initializing new one")
            self.initialize_model()
            return False
        
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            metadata = joblib.load(metadata_path)
            self.last_update = metadata['last_update']
            self.total_samples = metadata['total_samples']
            print(f"Loaded model for {self.item_id} (last update: {self.last_update}, samples: {self.total_samples})")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.initialize_model()
            return False
    
    def save_model(self):
        """Save model to disk."""
        joblib.dump(self.model, self._get_model_path())
        joblib.dump(self.scaler, self._get_scaler_path())
        metadata = {
            'last_update': self.last_update,
            'total_samples': self.total_samples
        }
        joblib.dump(metadata, self._get_metadata_path())
        print(f"Saved model for {self.item_id} (samples: {self.total_samples})")
    
    def prepare_features(self, entry):
        """Extract features from a bazaar data entry."""
        if self.mayor_data is None:
            self.mayor_data = get_mayor_perks()
        
        timestamp = entry.get('timestamp')
        if not timestamp:
            return None, None
        
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
            mayor_perks = match_mayor_perks(timestamp, self.mayor_data)
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
            
            return np.array(feature_vector), buy_price
            
        except (ValueError, TypeError) as e:
            return None, None
    
    def partial_fit(self, new_data):
        """
        Incrementally update model with new data.
        
        Args:
            new_data: List of bazaar data entries
            
        Returns:
            Number of samples processed
        """
        if self.model is None:
            self.initialize_model()
        
        features_list = []
        targets_list = []
        
        for entry in new_data:
            feature_vec, target = self.prepare_features(entry)
            if feature_vec is not None:
                features_list.append(feature_vec)
                targets_list.append(target)
        
        if len(features_list) == 0:
            print(f"No valid data to update model for {self.item_id}")
            return 0
        
        X = np.array(features_list)
        y = np.array(targets_list)
        
        # Update scaler incrementally
        if self.total_samples == 0:
            # First batch - fit scaler
            X_scaled = self.scaler.fit_transform(X)
        else:
            # Subsequent batches - partial fit scaler
            self.scaler.partial_fit(X)
            X_scaled = self.scaler.transform(X)
        
        # Update model
        self.model.partial_fit(X_scaled, y)
        
        self.total_samples += len(X)
        self.last_update = datetime.now()
        
        print(f"Updated {self.item_id} model with {len(X)} new samples (total: {self.total_samples})")
        return len(X)
    
    def predict(self, features):
        """
        Make prediction for given features.
        
        Args:
            features: Feature vector or array of feature vectors
            
        Returns:
            Predicted price(s)
        """
        if self.model is None:
            raise ValueError("Model not initialized or loaded")
        
        features = np.array(features)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)
    
    def predict_from_entry(self, entry):
        """
        Make prediction directly from a bazaar data entry.
        
        Args:
            entry: Bazaar data entry dict
            
        Returns:
            Predicted price or None if invalid entry
        """
        feature_vec, _ = self.prepare_features(entry)
        if feature_vec is None:
            return None
        
        return self.predict(feature_vec)[0]


class MultiItemIncrementalManager:
    """
    Manages multiple incremental models for different bazaar items.
    """
    
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.models = {}
        os.makedirs(model_dir, exist_ok=True)
    
    def get_or_create_model(self, item_id):
        """Get existing model or create new one for item."""
        if item_id not in self.models:
            model = IncrementalBazaarModel(item_id, self.model_dir)
            model.load_model()
            self.models[item_id] = model
        return self.models[item_id]
    
    def update_item(self, item_id, new_data):
        """Update model for specific item with new data."""
        model = self.get_or_create_model(item_id)
        samples_processed = model.partial_fit(new_data)
        model.save_model()
        return samples_processed
    
    def predict_item(self, item_id, entry):
        """Get prediction for specific item."""
        model = self.get_or_create_model(item_id)
        return model.predict_from_entry(entry)
    
    def save_all(self):
        """Save all models to disk."""
        for item_id, model in self.models.items():
            model.save_model()
    
    def get_model_status(self, item_id):
        """Get status information for a model."""
        if item_id not in self.models:
            return None
        
        model = self.models[item_id]
        return {
            'item_id': item_id,
            'total_samples': model.total_samples,
            'last_update': model.last_update.isoformat() if model.last_update else None
        }
