import requests
from datetime import datetime
from train_model import load_item_model, get_mayor_perks, match_mayor_perks
import numpy as np


class TradeRecommender:
    """
    Generates profitable trade recommendations based on model predictions.
    """
    
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.loaded_models = {}
        self.mayor_data = None
        
    def get_current_bazaar_data(self, item_id):
        """Fetch current bazaar data for an item from API."""
        url = f"https://sky.coflnet.com/api/bazaar/{item_id}"
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API error for {item_id}: {response.status_code}")
                return None
        except Exception as e:
            print(f"Failed to fetch {item_id}: {e}")
            return None
    
    def load_model_for_item(self, item_id):
        """Load model for specific item (with caching)."""
        if item_id in self.loaded_models:
            return self.loaded_models[item_id]
        
        model, scaler, metadata = load_item_model(item_id, self.model_dir)
        
        if model is None:
            return None
        
        self.loaded_models[item_id] = (model, scaler, metadata)
        return (model, scaler, metadata)
    
    def prepare_features_from_current_data(self, current_data):
        """Prepare feature vector from current bazaar data."""
        if self.mayor_data is None:
            self.mayor_data = get_mayor_perks()
        
        timestamp = current_data.get('timestamp', datetime.now().isoformat())
        
        try:
            # Current prices and volumes
            buy_price = float(current_data.get('buy', 0))
            sell_price = float(current_data.get('sell', 0))
            buy_volume = float(current_data.get('buyVolume', 0))
            sell_volume = float(current_data.get('sellVolume', 0))
            buy_moving_week = float(current_data.get('buyMovingWeek', 0))
            sell_moving_week = float(current_data.get('sellMovingWeek', 0))
            max_buy = float(current_data.get('maxBuy', 0))
            max_sell = float(current_data.get('maxSell', 0))
            min_buy = float(current_data.get('minBuy', 0))
            min_sell = float(current_data.get('minSell', 0))
            
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
            
            return np.array(feature_vector), buy_price, sell_price
            
        except (ValueError, TypeError, KeyError) as e:
            print(f"Error preparing features: {e}")
            return None, None, None
    
    def predict_price(self, item_id, current_data=None):
        """
        Predict future price for an item.
        
        Args:
            item_id: Bazaar item ID
            current_data: Current bazaar data (fetched if not provided)
            
        Returns:
            dict with prediction info or None if failed
        """
        # Load model
        model_info = self.load_model_for_item(item_id)
        if model_info is None:
            return None
        
        model, scaler, metadata = model_info
        
        # Get current data if not provided
        if current_data is None:
            current_data = self.get_current_bazaar_data(item_id)
            if current_data is None:
                return None
        
        # Prepare features
        features, current_buy, current_sell = self.prepare_features_from_current_data(current_data)
        if features is None:
            return None
        
        # Make prediction
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)
        predicted_price = model.predict(features_scaled)[0]
        
        return {
            'item_id': item_id,
            'current_buy_price': current_buy,
            'current_sell_price': current_sell,
            'predicted_price': predicted_price,
            'price_change': predicted_price - current_buy,
            'price_change_pct': ((predicted_price - current_buy) / current_buy * 100) if current_buy > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_profit_opportunity(self, item_id, current_data=None, capital=1000000):
        """
        Calculate profit opportunity for an item.
        
        Args:
            item_id: Bazaar item ID
            current_data: Current bazaar data
            capital: Available capital for trade
            
        Returns:
            dict with profit analysis or None
        """
        prediction = self.predict_price(item_id, current_data)
        if prediction is None:
            return None
        
        current_buy = prediction['current_buy_price']
        current_sell = prediction['current_sell_price']
        predicted_price = prediction['predicted_price']
        
        # Calculate if we should buy now and sell later
        if current_buy <= 0:
            return None
        
        # How many units can we buy?
        units_can_buy = capital / current_buy
        
        # Expected profit if we buy now and sell at predicted price
        expected_revenue = units_can_buy * predicted_price
        expected_profit = expected_revenue - capital
        expected_profit_pct = (expected_profit / capital) * 100
        
        # Instant flip profit (buy and immediately sell)
        instant_revenue = units_can_buy * current_sell
        instant_profit = instant_revenue - capital
        instant_profit_pct = (instant_profit / capital) * 100
        
        return {
            'item_id': item_id,
            'action': 'BUY' if expected_profit > 0 else 'HOLD',
            'current_buy_price': current_buy,
            'current_sell_price': current_sell,
            'predicted_price': predicted_price,
            'price_change_pct': prediction['price_change_pct'],
            'units_can_buy': units_can_buy,
            'expected_profit': expected_profit,
            'expected_profit_pct': expected_profit_pct,
            'instant_flip_profit': instant_profit,
            'instant_flip_profit_pct': instant_profit_pct,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_top_opportunities(self, item_ids, top_n=10, capital=1000000, min_profit_pct=1.0):
        """
        Get top N profitable trading opportunities.
        
        Args:
            item_ids: List of item IDs to analyze
            top_n: Number of top opportunities to return
            capital: Available capital per trade
            min_profit_pct: Minimum profit percentage threshold
            
        Returns:
            List of trade opportunities sorted by expected profit
        """
        opportunities = []
        
        print(f"Analyzing {len(item_ids)} items for trading opportunities...")
        
        for item_id in item_ids:
            opportunity = self.calculate_profit_opportunity(item_id, capital=capital)
            
            if opportunity and opportunity['expected_profit_pct'] >= min_profit_pct:
                opportunities.append(opportunity)
        
        # Sort by expected profit percentage (descending)
        opportunities.sort(key=lambda x: x['expected_profit_pct'], reverse=True)
        
        return opportunities[:top_n]
    
    def format_recommendation(self, opportunity):
        """Format a trading opportunity as readable text."""
        item = opportunity['item_id']
        action = opportunity['action']
        buy_price = opportunity['current_buy_price']
        predicted = opportunity['predicted_price']
        profit_pct = opportunity['expected_profit_pct']
        profit_coins = opportunity['expected_profit']
        units = opportunity['units_can_buy']
        
        return (
            f"{item}: {action}\n"
            f"  Buy at: {buy_price:,.2f} coins\n"
            f"  Predicted: {predicted:,.2f} coins ({profit_pct:+.2f}%)\n"
            f"  Units: {units:,.0f}\n"
            f"  Expected Profit: {profit_coins:+,.0f} coins"
        )


def get_all_bazaar_items():
    """Fetch list of all bazaar item IDs."""
    try:
        response = requests.get("https://sky.coflnet.com/api/items/bazaar/tags", timeout=5)
        if response.status_code == 200:
            import re
            item_ids = re.findall(r'"([^"]*)"', response.text)
            return item_ids
        else:
            return []
    except Exception as e:
        print(f"Error fetching item list: {e}")
        return []


if __name__ == "__main__":
    # Example usage
    recommender = TradeRecommender()
    
    # Get all items
    item_ids = get_all_bazaar_items()
    print(f"Found {len(item_ids)} bazaar items")
    
    # Get top opportunities
    opportunities = recommender.get_top_opportunities(
        item_ids[:20],  # Test with first 20 items
        top_n=5,
        capital=10000000,  # 10M coins
        min_profit_pct=2.0  # Minimum 2% profit
    )
    
    print(f"\n{'='*60}")
    print(f"TOP {len(opportunities)} TRADING OPPORTUNITIES")
    print(f"{'='*60}\n")
    
    for i, opp in enumerate(opportunities, 1):
        print(f"{i}. {recommender.format_recommendation(opp)}\n")
