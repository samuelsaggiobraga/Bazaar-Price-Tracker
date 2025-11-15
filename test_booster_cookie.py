import json
import os
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from data_utils import load_or_fetch_item_data
from mayor_utils import get_mayor_perks, match_mayor_perks
from outlier_utils import remove_outliers
from backtest import run_backtest, print_backtest_report


def prepare_features(timestamp, buy_price, sell_price, buy_volume, sell_volume,
                    buy_moving_week, sell_moving_week, max_buy, max_sell,
                    min_buy, min_sell, mayor_data):
    """Prepare feature vector from raw data."""
    year = int(timestamp[:4])
    month = int(timestamp[5:7])
    day = int(timestamp[8:10])
    
    date_obj = datetime.strptime(timestamp[:10], '%Y-%m-%d')
    day_of_year = date_obj.timetuple().tm_yday
    day_of_week = date_obj.weekday()
    
    mayor_perks = match_mayor_perks(timestamp, mayor_data)
    
    feature_vector = [
        year, month, day, day_of_year, day_of_week,
        buy_price, sell_price,
        buy_volume, sell_volume,
        buy_moving_week, sell_moving_week,
        max_buy, max_sell,
        min_buy, min_sell
    ] + mayor_perks
    
    return feature_vector




def train_and_test_single_item(item_id, cleanup=False):
    """Train model on single item and generate predictions.
    
    Args:
        item_id: The bazaar item ID to train on
        cleanup: If True, delete the JSON data file after training
    """
    print(f"Training model on {item_id}...")
    
    # Fetch mayor data
    mayor_data = get_mayor_perks()
    print(f"Loaded {len(mayor_data)} mayor periods")
    
    # Load item data
    data = load_or_fetch_item_data(item_id)
    if data is None:
        return
    
    # Parse data
    features = []
    targets = []
    timestamps = []
    
    for entry in data:
        if not isinstance(entry, dict):
            continue
        
        timestamp = entry.get('timestamp')
        if not timestamp:
            continue
        
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
            
            feature_vector = prepare_features(
                timestamp, buy_price, sell_price, buy_volume, sell_volume,
                buy_moving_week, sell_moving_week, max_buy, max_sell,
                min_buy, min_sell, mayor_data
            )
            
            features.append(feature_vector)
            targets.append(buy_price)
            timestamps.append(timestamp)
            
        except (ValueError, TypeError):
            continue
    
    X = np.array(features)
    y = np.array(targets)
    timestamps = np.array(timestamps)
    
    print(f"Total data points (before outlier removal): {len(X)}")
    
    # Remove outliers using Tukey's method
    X, y, timestamps, outlier_info = remove_outliers(X, y, timestamps, method='tukey', multiplier=1.5)
    
    print(f"Outliers removed: {outlier_info['outliers_removed']} ({outlier_info['outlier_percentage']:.2f}%)")
    print(f"Training data points (after outlier removal): {len(X)}")
    print(f"Outlier bounds: [{outlier_info['lower_bound']:.2f}, {outlier_info['upper_bound']:.2f}]")
    
    # Find start date from data
    dates = [datetime.strptime(ts[:10], '%Y-%m-%d') for ts in timestamps]
    start_date = min(dates)
    
    # Split into train/test based on time (85/15 split)
    # Sort by date to ensure chronological split
    date_indices = np.argsort([datetime.strptime(ts[:10], '%Y-%m-%d') for ts in timestamps])
    X = X[date_indices]
    y = y[date_indices]
    timestamps = timestamps[date_indices]
    
    # Calculate 85/15 split point
    split_idx = int(len(X) * 0.85)
    split_date = datetime.strptime(timestamps[split_idx][:10], '%Y-%m-%d')
    today = datetime.now()
    
    # Create train/test sets
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    timestamps_train, timestamps_test = timestamps[:split_idx], timestamps[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining Random Forest...")
    model = RandomForestRegressor(
        n_estimators=50,  # Reduced from 100
        max_depth=10,  # Reduced from 20 to prevent overfitting
        min_samples_split=20,  # Increased from 5 to require more samples per split
        min_samples_leaf=10,  # Increased from 2 to require more samples per leaf
        max_features='sqrt',  # Use sqrt of features to add randomness
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions on test set
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\n{'='*50}")
    print(f"ACCURACY METRICS FOR {item_id}")
    print(f"{'='*50}")
    print(f"Mean Absolute Error: {mae:.2f} coins")
    print(f"Root Mean Squared Error: {rmse:.2f} coins")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"{'='*50}\n")
    
    # Generate future predictions (next 30 days)
    future_dates = [today + timedelta(days=i) for i in range(1, 31)]
    
    # Use last known values for volume/moving week features
    last_entry = data[-1]
    last_sell_price = float(last_entry.get('sell', y_test[-1] if len(y_test) > 0 else y_train[-1]))
    last_buy_volume = float(last_entry.get('buyVolume', 0))
    last_sell_volume = float(last_entry.get('sellVolume', 0))
    last_buy_moving = float(last_entry.get('buyMovingWeek', 0))
    last_sell_moving = float(last_entry.get('sellMovingWeek', 0))
    last_max_buy = float(last_entry.get('maxBuy', 0))
    last_max_sell = float(last_entry.get('maxSell', 0))
    last_min_buy = float(last_entry.get('minBuy', 0))
    last_min_sell = float(last_entry.get('minSell', 0))
    
    future_features = []
    for future_date in future_dates:
        timestamp_str = future_date.strftime('%Y-%m-%d')
        
        # Use last predicted price or last known price
        pred_price = y_pred[-1] if len(y_pred) > 0 else y_train[-1]
        
        feature_vector = prepare_features(
            timestamp_str, pred_price, last_sell_price,
            last_buy_volume, last_sell_volume,
            last_buy_moving, last_sell_moving,
            last_max_buy, last_max_sell,
            last_min_buy, last_min_sell,
            mayor_data
        )
        future_features.append(feature_vector)
    
    future_features = np.array(future_features)
    future_features_scaled = scaler.transform(future_features)
    future_predictions = model.predict(future_features_scaled)
    
    # Prepare data for visualization
    all_dates = [datetime.strptime(ts[:10], '%Y-%m-%d') for ts in timestamps]
    test_dates_dt = [datetime.strptime(ts[:10], '%Y-%m-%d') for ts in timestamps_test]
    
    current_price = y_test[-1] if len(y_test) > 0 else y_train[-1]
    avg_future = np.mean(future_predictions)
    
    # Print forecast roadmap to console
    print(f"\n{'='*60}")
    print(f"30-DAY PRICE FORECAST ROADMAP FOR {item_id}")
    print(f"{'='*60}")
    print(f"{'Date':<15} {'Predicted Price':<20} {'Change from Today':<20}")
    print(f"{'-'*60}")
    
    for i, (date, price) in enumerate(zip(future_dates, future_predictions)):
        change = ((price - current_price) / current_price) * 100
        print(f"{date.strftime('%Y-%m-%d'):<15} {price:>10,.2f} coins     {change:>+7.2f}%")
        
        # Highlight weekly milestones
        if (i + 1) % 7 == 0:
            print(f"{'-'*60}")
    
    print(f"{'='*60}")
    
    trend = "UPWARD" if avg_future > current_price else "DOWNWARD"
    magnitude = abs(((avg_future - current_price) / current_price) * 100)
    
    print(f"\nForecast Summary:")
    print(f"  Current Price: {current_price:,.2f} coins")
    print(f"  Average 30-day Forecast: {avg_future:,.2f} coins")
    print(f"  Expected Trend: {trend} ({magnitude:.2f}%)")
    print(f"  Minimum Expected: {np.min(future_predictions):,.2f} coins")
    print(f"  Maximum Expected: {np.max(future_predictions):,.2f} coins")
    
    # Run backtest simulation
    print(f"\nRunning backtest simulation...")
    backtest_results = run_backtest(X_test, y_test, model, scaler, timestamps_test)
    print_backtest_report(backtest_results)
    
    # Save model
    joblib.dump(model, f'{item_id}_model.pkl')
    joblib.dump(scaler, f'{item_id}_scaler.pkl')
    print(f"\nModel saved as {item_id}_model.pkl")
    
    # Cleanup JSON file if requested
    if cleanup:
        filename = f"bazaar_history_combined_{item_id}.json"
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Cleaned up {filename}")


if __name__ == "__main__":
    train_and_test_single_item("BOOSTER_COOKIE")
