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


def train_model_on_items(item_ids, delete_files=True):
    """Train a model on multiple items."""
    print("Fetching mayor perks data...")
    mayor_data = get_mayor_perks()
    print(f"Loaded {len(mayor_data)} mayor periods")
    
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


def save_model(model, scaler, model_name="bazaar_price_model"):
    """Save trained model and scaler."""
    joblib.dump(model, f"{model_name}.pkl")
    joblib.dump(scaler, f"{model_name}_scaler.pkl")
    print(f"Model saved as {model_name}.pkl")
    print(f"Scaler saved as {model_name}_scaler.pkl")


def load_model(model_name="bazaar_price_model"):
    """Load a trained model and scaler."""
    model = joblib.load(f"{model_name}.pkl")
    scaler = joblib.load(f"{model_name}_scaler.pkl")
    return model, scaler


if __name__ == "__main__":
    # Example: Get all bazaar item IDs
    import requests
    temp = requests.get("https://sky.coflnet.com/api/items/bazaar/tags")
    item_ids = re.findall(r'"([^"]*)"', temp.text)
    
    print(f"Found {len(item_ids)} items to train on")
    
    # Train on all items (will delete files as processed)
    model, scaler = train_model_on_items(item_ids, delete_files=True)
    
    if model is not None:
        save_model(model, scaler)
