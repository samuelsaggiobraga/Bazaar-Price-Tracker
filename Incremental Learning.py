import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from Bazaar import get_bazaar_buy_data, get_bazaar_sell_data
import json
import time
import os
import joblib
from collections import deque
from datetime import datetime
from ZScore import calculate_z_score_outliers, calculate_tukey_outliers
from hypixel_api_lib import Elections
import numpy as np
from Mayor import get_perk_encoding
 

MODEL_PATH = 'bazaar_model.joblib'
HISTORY_PATH = 'price_history.json'
SAMPLE_INTERVAL = 300  

class PriceRecord:
    def __init__(self, price, timestamp):
        self.price = price
        self.timestamp = timestamp

    def to_dict(self):
        return {"price": self.price, "timestamp": self.timestamp}

    @staticmethod
    def from_dict(d):
        return PriceRecord(d["price"], d["timestamp"])

def load_or_create_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'r') as f:
            history_data = json.load(f)
            history = deque(maxlen=1000)
            for item in history_data:
                history.append(PriceRecord.from_dict(item))
            return history
    return deque(maxlen=1000)

def save_history(price_history):
    with open(HISTORY_PATH, 'w') as f:
        history_list = [record.to_dict() for record in price_history]
        json.dump(history_list, f)

def load_or_create_model():
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        return joblib.load(MODEL_PATH)
    print("Creating new model...")
    return SGDClassifier(
        warm_start=True,
        learning_rate='adaptive',
        eta0=0.01,
        random_state=42
    )

def IncrementalLearningModel():
    price_history = load_or_create_history()
    model = load_or_create_model()
    scaler = StandardScaler()
    
    if not price_history:
        perk_len = len(get_perk_encoding())
        feature_dim = 1 + perk_len 
        dummy_X = np.zeros((2, feature_dim), dtype=np.float64)
        dummy_X[1, 0] = 1.0  
        dummy_y = np.array([0, 1])
        model.partial_fit(dummy_X, dummy_y, classes=np.array([0, 1]))

    print("Starting price tracking and model training...")
    print(f"Sampling interval: {SAMPLE_INTERVAL} seconds ({SAMPLE_INTERVAL/60:.1f} minutes)")
    print(f"Current history size: {len(price_history)} prices")
    print("Press Ctrl+C to stop.")
    
    last_save_time = time.time()
    
    try:
        while True:
            current_time = time.time()
            new_buy_price = get_bazaar_buy_data("BOOSTER_COOKIE")
              
            
            if new_buy_price is not None:
                new_record = PriceRecord(new_buy_price, datetime.now().isoformat())
                price_history.append(new_record)
                
                if len(price_history) >= 2:
                    prices_array = np.array([record.price for record in price_history])
                    
                    rolling_mean = np.mean(prices_array)
                    rolling_std = np.std(prices_array)
                    
                    price_change = 1 if new_buy_price > price_history[-2].price else 0
                    
                    if rolling_std != 0:
                        z_score = (new_buy_price - rolling_mean) / rolling_std
                    else:
                        z_score = 0
                    
                    perk_vector = np.asarray(get_perk_encoding())
                    z_arr = np.array([z_score])
                    X_new = np.hstack((z_arr, perk_vector)).astype(np.float64).reshape(1, -1)
                    y_new = np.array([price_change])
                    
                    
                    
                    try:
                        pred_before = model.predict(X_new)[0]
                    except:
                        pred_before = None
                    
                    model.partial_fit(X_new, y_new)
                    pred_after = model.predict(X_new)[0]
                    
                    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Current Price: {new_buy_price:,.2f}")
                    print(f"Previous Price: {price_history[-2].price:,.2f}")
                    print(f"Rolling Mean: {rolling_mean:,.2f}")
                    print(f"Rolling Std: {rolling_std:,.2f}")
                    print(f"Z-score: {z_score:.2f}")
                    print(f"Price Change Direction: {'Up' if price_change == 1 else 'Down'}")
                    if pred_before is not None:
                        print(f"Model Prediction (before update): {'Up' if pred_before == 1 else 'Down'}")
                    print(f"Model Prediction (after update): {'Up' if pred_after == 1 else 'Down'}")
                    
                    # Save model and history every 15 minutes
                    if current_time - last_save_time > 900:  # 900 seconds = 15 minutes
                        print("\nSaving model and history...")
                        joblib.dump(model, MODEL_PATH)
                        save_history(price_history)
                        last_save_time = current_time
                        print("Save complete.")
                    
                else:
                    print(f"\nCollecting initial price data: {new_buy_price:,.2f}")
            
            # Sleep until next sample time
            time.sleep(SAMPLE_INTERVAL)
    
    except KeyboardInterrupt:
        print("\nPrice tracking stopped.")
        
        # Save final state
        print("\nSaving final model and history...")
        joblib.dump(model, MODEL_PATH)
        save_history(price_history)
        print("Save complete.")
        
        if len(price_history) >= 2:
            prices_array = np.array([record.price for record in price_history])
            times_array = [record.timestamp for record in price_history]
            
            print(f"\nFinal Statistics:")
            print(f"Total price points collected: {len(price_history)}")
            print(f"Data collection period: {times_array[0]} to {times_array[-1]}")
            print(f"Price range: {min(prices_array):,.2f} - {max(prices_array):,.2f}")
            print(f"Mean price: {np.mean(prices_array):,.2f}")
            print(f"Standard deviation: {np.std(prices_array):,.2f}")
            print(f"\nModel and price history saved to:")
            print(f"- Model: {MODEL_PATH}")
            print(f"- History: {HISTORY_PATH}")

if __name__ == "__main__":
    IncrementalLearningModel()