import json
import os
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from data_utils import load_or_fetch_item_data
from mayor_utils import get_mayor_perks, match_mayor_perks


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
    
    print(f"Total data points: {len(X)}")
    
    # Find start date from data
    dates = [datetime.strptime(ts[:10], '%Y-%m-%d') for ts in timestamps]
    start_date = min(dates)
    
    # Split into train/test based on time
    today = datetime.now()
    test_days = 30  # Last 30 days for testing
    split_date = today - timedelta(days=test_days)
    
    train_mask = np.array([datetime.strptime(ts[:10], '%Y-%m-%d') < split_date for ts in timestamps])
    test_mask = ~train_mask
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    timestamps_train, timestamps_test = timestamps[train_mask], timestamps[test_mask]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
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
    
    # Create comprehensive plot
    plt.figure(figsize=(16, 10))
    plt.style.use('dark_background')
    
    # Plot 1: Historical data with predictions
    plt.subplot(2, 1, 1)
    
    # All historical dates
    all_dates = [datetime.strptime(ts[:10], '%Y-%m-%d') for ts in timestamps]
    
    plt.plot(all_dates, y, 'o-', color='#4A5568', alpha=0.6, markersize=2, 
             label='Historical Actual', linewidth=1)
    
    # Test predictions
    test_dates = [datetime.strptime(ts[:10], '%Y-%m-%d') for ts in timestamps_test]
    plt.plot(test_dates, y_test, 'o', color='#F7FAFC', markersize=4, 
             label='Test Actual', zorder=5)
    plt.plot(test_dates, y_pred, 's', color='#718096', markersize=4, 
             label='Test Predicted', zorder=5)
    
    # Future predictions
    plt.plot(future_dates, future_predictions, 'D-', color='#2D3748', 
             markersize=5, linewidth=2, label='Future Forecast', zorder=10)
    
    plt.axvline(x=split_date, color='#1A202C', linestyle='--', linewidth=2, 
                label='Train/Test Split')
    plt.axvline(x=today, color='#2D3748', linestyle='--', linewidth=2, 
                label='Today')
    
    plt.xlabel('Date', fontsize=12, color='#E2E8F0')
    plt.ylabel('Buy Price (coins)', fontsize=12, color='#E2E8F0')
    plt.title(f'{item_id} - Price Prediction Model (Start: {start_date.strftime("%Y-%m-%d")} to Today + 30 days)',
              fontsize=14, fontweight='bold', color='#F7FAFC')
    plt.legend(loc='best', facecolor='#1A202C', edgecolor='#2D3748')
    plt.grid(True, alpha=0.2, color='#4A5568')
    plt.xticks(rotation=45, color='#CBD5E0')
    plt.yticks(color='#CBD5E0')
    
    # Plot 2: Forecast details
    plt.subplot(2, 1, 2)
    
    forecast_days = list(range(1, 31))
    plt.plot(forecast_days, future_predictions, 'D-', color='#2D3748', 
             markersize=6, linewidth=2)
    plt.fill_between(forecast_days, future_predictions * 0.95, future_predictions * 1.05,
                      alpha=0.2, color='#4A5568')
    
    plt.xlabel('Days from Today', fontsize=12, color='#E2E8F0')
    plt.ylabel('Predicted Price (coins)', fontsize=12, color='#E2E8F0')
    plt.title('30-Day Price Forecast with 5% Confidence Band', 
              fontsize=14, fontweight='bold', color='#F7FAFC')
    plt.grid(True, alpha=0.2, color='#4A5568')
    plt.xticks(color='#CBD5E0')
    plt.yticks(color='#CBD5E0')
    
    # Add accuracy text box
    textstr = f'Model Accuracy:\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.4f}\nMAPE: {mape:.2f}%'
    props = dict(boxstyle='round', facecolor='#1A202C', edgecolor='#2D3748', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props, color='#E2E8F0')
    
    plt.tight_layout()
    plt.savefig(f'{item_id}_forecast.png', dpi=150, facecolor='#0F1419')
    print(f"Graph saved as {item_id}_forecast.png")
    plt.show()
    
    # Print forecast roadmap
    print(f"\n{'='*60}")
    print(f"30-DAY PRICE FORECAST ROADMAP FOR {item_id}")
    print(f"{'='*60}")
    print(f"{'Date':<15} {'Predicted Price':<20} {'Change from Today':<20}")
    print(f"{'-'*60}")
    
    current_price = y_test[-1] if len(y_test) > 0 else y_train[-1]
    
    for i, (date, price) in enumerate(zip(future_dates, future_predictions)):
        change = ((price - current_price) / current_price) * 100
        print(f"{date.strftime('%Y-%m-%d'):<15} {price:>10,.2f} coins     {change:>+7.2f}%")
        
        # Highlight weekly milestones
        if (i + 1) % 7 == 0:
            print(f"{'-'*60}")
    
    print(f"{'='*60}")
    
    avg_future = np.mean(future_predictions)
    trend = "UPWARD" if avg_future > current_price else "DOWNWARD"
    magnitude = abs(((avg_future - current_price) / current_price) * 100)
    
    print(f"\nForecast Summary:")
    print(f"  Current Price: {current_price:,.2f} coins")
    print(f"  Average 30-day Forecast: {avg_future:,.2f} coins")
    print(f"  Expected Trend: {trend} ({magnitude:.2f}%)")
    print(f"  Minimum Expected: {np.min(future_predictions):,.2f} coins")
    print(f"  Maximum Expected: {np.max(future_predictions):,.2f} coins")
    
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
