import json
import os
import numpy as np
from datetime import datetime
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import requests


def get_mayor_perks():
    """Fetch mayor perks data and return as binary vectors with timestamps."""
    current_datetime = datetime.now()
    url = f"https://sky.coflnet.com/api/mayor?from=2025-02-17T20%3A03%3A10.937Z&to={current_datetime.strftime('%Y-%m-%dT%H%%3A%M%%3A%S.%fZ')}"
    
    response = requests.get(url)
    html_content = response.text
    mayors = html_content.split('"start"')
    mayors.pop(0)
    
    mayor_data = []
    for mayor in mayors:
        # Extract start timestamp
        start_match = re.search(r'(\d{4}-\d{2}-\d{2})', mayor)
        if not start_match:
            continue
        start_date = datetime.strptime(start_match.group(1), '%Y-%m-%d')
        
        # Extract perks as binary vector
        binary_perks = [0 for _ in range(40)]
        matches = re.findall(r'"name":"([^"]*)"', mayor)
        
        with open("perk_names.txt", "r") as f:
            perk_names = f.read().splitlines()
        
        for perk_name in matches:
            if perk_name in perk_names:
                perk_index = perk_names.index(perk_name)
                binary_perks[perk_index] = 1
        
        mayor_data.append({
            'start_date': start_date,
            'perks': binary_perks
        })
    
    return mayor_data


def match_mayor_perks(timestamp_str, mayor_data):
    """Match a timestamp to the appropriate mayor perks, or return None if too old."""
    try:
        data_date = datetime.strptime(timestamp_str[:10], '%Y-%m-%d')
    except:
        return None
    
    # Find the mayor in office at this timestamp
    for i, mayor in enumerate(mayor_data):
        if i + 1 < len(mayor_data):
            if mayor['start_date'] <= data_date < mayor_data[i + 1]['start_date']:
                return mayor['perks']
        else:
            # Last mayor entry
            if mayor['start_date'] <= data_date:
                return mayor['perks']
    
    return None


def load_and_prepare_data(item_id, mayor_data, delete_after=True):
    """Load item data, prepare features and targets."""
    filename = f"bazaar_history_combined_{item_id}.json"
    
    if not os.path.exists(filename):
        print(f"File {filename} not found, skipping...")
        return None, None
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    if len(data) == 0:
        if delete_after:
            os.remove(filename)
            print(f"Deleted empty file: {filename}")
        return None, None
    
    features = []
    targets = []
    timestamps = []
    
    for entry in data:
        if not isinstance(entry, dict):
            continue
        
        timestamp = entry.get('timestamp')
        if not timestamp:
            continue
        
        # Extract base features
        try:
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
            
            # Extract date features
            year = int(timestamp[:4])
            month = int(timestamp[5:7])
            day = int(timestamp[8:10])
            
            # Day of year (seasonal feature)
            date_obj = datetime.strptime(timestamp[:10], '%Y-%m-%d')
            day_of_year = date_obj.timetuple().tm_yday
            day_of_week = date_obj.weekday()
            
        except (ValueError, TypeError):
            continue
        
        # Match mayor perks
        mayor_perks = match_mayor_perks(timestamp, mayor_data)
        if mayor_perks is None:
            # No mayor data available for this timestamp - use zeros
            mayor_perks = [0] * 40
        
        # Combine all features
        feature_vector = [
            year, month, day, day_of_year, day_of_week,
            buy_price, sell_price,
            buy_volume, sell_volume,
            buy_moving_week, sell_moving_week,
            max_buy, max_sell,
            min_buy, min_sell
        ] + mayor_perks
        
        features.append(feature_vector)
        targets.append(buy_price)  # Predicting buy price
        timestamps.append(timestamp)
    
    if delete_after:
        os.remove(filename)
        print(f"Deleted processed file: {filename}")
    
    return np.array(features), np.array(targets), timestamps


def train_single_item_model(item_id, mayor_data, delete_file=False):
    """Train a model on a single item.
    
    Args:
        item_id: The bazaar item ID to train on
        mayor_data: Mayor perks data
        delete_file: If True, delete the JSON file after training
        
    Returns:
        tuple: (model, scaler, metadata_dict) or (None, None, None) if failed
    """
    print(f"\nTraining model for {item_id}...")
    result = load_and_prepare_data(item_id, mayor_data, delete_after=delete_file)
    
    if result[0] is None or len(result) != 3:
        return None, None, None
    
    features, targets, timestamps = result
    
    if len(features) == 0:
        print(f"  No data for {item_id}")
        return None, None, None
    
    print(f"  Loaded {len(features)} data points")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, targets)
    
    # Metadata for tracking
    metadata = {
        'item_id': item_id,
        'total_samples': len(features),
        'trained_at': datetime.now().isoformat(),
        'feature_count': features.shape[1]
    }
    
    print(f"  Training complete for {item_id}")
    
    return model, scaler, metadata


def train_model_on_items(item_ids, delete_files=True, per_item=True):
    """Train models on multiple items.
    
    Args:
        item_ids: List of item IDs to train on
        delete_files: If True, delete JSON files after training
        per_item: If True, create separate model per item. If False, combine all items.
        
    Returns:
        If per_item=True: dict mapping item_id to (model, scaler, metadata)
        If per_item=False: (model, scaler) for combined model
    """
    print("Fetching mayor perks data...")
    mayor_data = get_mayor_perks()
    print(f"Loaded {len(mayor_data)} mayor periods")
    
    if per_item:
        # Train separate model for each item
        models_dict = {}
        
        for item_id in item_ids:
            model, scaler, metadata = train_single_item_model(item_id, mayor_data, delete_file=delete_files)
            if model is not None:
                models_dict[item_id] = (model, scaler, metadata)
        
        print(f"\nTrained {len(models_dict)} models successfully")
        return models_dict
    
    else:
        # Original behavior: combine all items into one model
        all_features = []
        all_targets = []
        
        for item_id in item_ids:
            print(f"\nProcessing {item_id}...")
            features, targets, _ = load_and_prepare_data(item_id, mayor_data, delete_after=delete_files)
            
            if features is not None and len(features) > 0:
                all_features.append(features)
                all_targets.append(targets)
                print(f"  Loaded {len(features)} data points")
        
        if len(all_features) == 0:
            print("No data to train on!")
            return None, None
        
        # Combine all data
        X = np.vstack(all_features)
        y = np.hstack(all_targets)
        
        print(f"\nTotal training samples: {len(X)}")
        print(f"Feature dimensions: {X.shape[1]}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        print("\nTraining Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_scaled, y)
        
        print("Training complete!")
        
        return model, scaler


def save_model(model, scaler, model_name="bazaar_price_model", metadata=None):
    """Save trained model and scaler."""
    joblib.dump(model, f"{model_name}.pkl")
    joblib.dump(scaler, f"{model_name}_scaler.pkl")
    if metadata is not None:
        joblib.dump(metadata, f"{model_name}_metadata.pkl")
    print(f"Model saved as {model_name}.pkl")
    print(f"Scaler saved as {model_name}_scaler.pkl")


def save_models(models_dict, output_dir="."):
    """Save multiple per-item models.
    
    Args:
        models_dict: Dict mapping item_id to (model, scaler, metadata)
        output_dir: Directory to save models in
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for item_id, (model, scaler, metadata) in models_dict.items():
        model_name = os.path.join(output_dir, item_id)
        save_model(model, scaler, model_name, metadata)
    
    print(f"\nSaved {len(models_dict)} models to {output_dir}")


def load_model(model_name="bazaar_price_model", load_metadata=False):
    """Load a trained model and scaler.
    
    Args:
        model_name: Base name of the model files (without .pkl extension)
        load_metadata: If True, also load and return metadata
        
    Returns:
        (model, scaler) if load_metadata=False
        (model, scaler, metadata) if load_metadata=True
    """
    model = joblib.load(f"{model_name}.pkl")
    scaler = joblib.load(f"{model_name}_scaler.pkl")
    
    if load_metadata:
        metadata_path = f"{model_name}_metadata.pkl"
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            return model, scaler, metadata
        else:
            return model, scaler, None
    
    return model, scaler


def load_item_model(item_id, model_dir="models"):
    """Load model for a specific item.
    
    Args:
        item_id: The bazaar item ID
        model_dir: Directory containing the models
        
    Returns:
        (model, scaler, metadata) or (None, None, None) if not found
    """
    model_path = os.path.join(model_dir, item_id)
    
    if not os.path.exists(f"{model_path}.pkl"):
        print(f"Model not found for {item_id}")
        return None, None, None
    
    return load_model(model_path, load_metadata=True)


if __name__ == "__main__":
    # Example: Get all bazaar item IDs
    import requests
    temp = requests.get("https://sky.coflnet.com/api/items/bazaar/tags")
    item_ids = re.findall(r'"([^"]*)"', temp.text)
    
    print(f"Found {len(item_ids)} items to train on")
    
    # Choose training mode
    PER_ITEM = True  # Set to False for single combined model (old behavior)
    
    if PER_ITEM:
        # Train separate model for each item (recommended for day trading)
        models_dict = train_model_on_items(item_ids, delete_files=False, per_item=True)
        if models_dict:
            save_models(models_dict, output_dir="models")
    else:
        # Train single model on all items combined (original behavior)
        model, scaler = train_model_on_items(item_ids, delete_files=True, per_item=False)
        if model is not None:
            save_model(model, scaler)
