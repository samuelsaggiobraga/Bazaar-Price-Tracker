"""
Trading Service - Background daemon with REST API for Minecraft mod integration.

Runs continuously to:
1. Fetch new bazaar data periodically
2. Update models incrementally
3. Generate trade recommendations
4. Serve API endpoints for Minecraft mod
"""

import time
import threading
import requests
import json
from datetime import datetime, timedelta
from trade_recommender import TradeRecommender, get_all_bazaar_items
from incremental_trainer import IncrementalTrainer
import os


class TradingService:
    """Background service for continuous trading analysis."""
    
    def __init__(
        self,
        model_dir="models",
        data_dir=".",
        fetch_interval=900,  # 15 minutes
        retrain_interval=3600,  # 1 hour
        sliding_window_days=30
    ):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.fetch_interval = fetch_interval
        self.retrain_interval = retrain_interval
        self.sliding_window_days = sliding_window_days
        
        self.recommender = TradeRecommender(model_dir)
        self.item_ids = []
        self.last_fetch = None
        self.last_retrain = {}
        self.running = False
        
        # Cache for latest opportunities
        self.cached_opportunities = []
        self.cache_timestamp = None
        
        # Service statistics
        self.stats = {
            'service_started': datetime.now().isoformat(),
            'total_fetches': 0,
            'total_retrains': 0,
            'items_tracked': 0,
            'last_update': None
        }
    
    def initialize(self):
        """Initialize service - load item list."""
        print("Initializing trading service...")
        self.item_ids = get_all_bazaar_items()
        self.stats['items_tracked'] = len(self.item_ids)
        print(f"Loaded {len(self.item_ids)} bazaar items")
        
        # Initialize last_retrain timestamps
        for item_id in self.item_ids:
            self.last_retrain[item_id] = datetime.now() - timedelta(hours=2)
    
    def fetch_current_data(self, item_id):
        """Fetch current bazaar data for an item."""
        url = f"https://sky.coflnet.com/api/bazaar/{item_id}"
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Add timestamp if not present
                if 'timestamp' not in data:
                    data['timestamp'] = datetime.now().isoformat()
                return data
            return None
        except Exception as e:
            print(f"Error fetching {item_id}: {e}")
            return None
    
    def fetch_all_current_data(self):
        """Fetch current data for all items."""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Fetching current bazaar data...")
        
        current_data = {}
        fetched = 0
        
        for item_id in self.item_ids:
            data = self.fetch_current_data(item_id)
            if data:
                current_data[item_id] = data
                fetched += 1
            
            # Rate limiting - be nice to the API
            if fetched % 10 == 0:
                time.sleep(0.5)
        
        self.last_fetch = datetime.now()
        self.stats['total_fetches'] += 1
        self.stats['last_update'] = self.last_fetch.isoformat()
        
        print(f"Fetched data for {fetched}/{len(self.item_ids)} items")
        return current_data
    
    def retrain_models(self, item_ids_to_retrain, current_data):
        """Retrain models for specific items."""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Retraining {len(item_ids_to_retrain)} models...")
        
        retrained = 0
        
        for item_id in item_ids_to_retrain:
            # Get new data entries for this item
            new_entries = []
            if item_id in current_data:
                new_entries = [current_data[item_id]]
            
            # Retrain
            trainer = IncrementalTrainer(item_id, self.model_dir, self.data_dir)
            
            # Load existing model
            loaded = trainer.load_existing_model()
            
            if not loaded:
                print(f"  Skipping {item_id} - no existing model")
                continue
            
            # Append new data
            if len(new_entries) > 0:
                trainer.append_new_data(new_entries)
            
            # Retrain with sliding window
            success = trainer.retrain_with_new_data(new_entries, self.sliding_window_days)
            
            if success:
                self.last_retrain[item_id] = datetime.now()
                retrained += 1
                self.stats['total_retrains'] += 1
        
        print(f"Successfully retrained {retrained} models")
    
    def update_cycle(self):
        """Single update cycle - fetch data and retrain if needed."""
        # Fetch current data
        current_data = self.fetch_all_current_data()
        
        # Determine which models need retraining
        now = datetime.now()
        items_to_retrain = []
        
        for item_id in self.item_ids:
            last_retrain_time = self.last_retrain.get(item_id)
            if last_retrain_time is None:
                continue
            
            time_since_retrain = (now - last_retrain_time).total_seconds()
            
            if time_since_retrain >= self.retrain_interval:
                items_to_retrain.append(item_id)
        
        # Retrain models that are due
        if len(items_to_retrain) > 0:
            # Limit retraining to avoid overload (retrain max 10 at a time)
            items_to_retrain = items_to_retrain[:10]
            self.retrain_models(items_to_retrain, current_data)
        
        # Update recommendations cache
        self.update_recommendations_cache()
    
    def update_recommendations_cache(self):
        """Update cached trading recommendations."""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Updating recommendations...")
        
        # Only analyze items that have models
        items_with_models = []
        for item_id in self.item_ids:
            model_path = os.path.join(self.model_dir, f"{item_id}.pkl")
            if os.path.exists(model_path):
                items_with_models.append(item_id)
        
        if len(items_with_models) == 0:
            print("No trained models available yet")
            return
        
        # Get top opportunities
        opportunities = self.recommender.get_top_opportunities(
            items_with_models,
            top_n=20,
            capital=10000000,
            min_profit_pct=1.0
        )
        
        self.cached_opportunities = opportunities
        self.cache_timestamp = datetime.now()
        
        print(f"Found {len(opportunities)} profitable opportunities")
    
    def run_background(self):
        """Run background update loop."""
        self.running = True
        print(f"\n{'='*60}")
        print("Trading Service Started")
        print(f"Fetch interval: {self.fetch_interval}s ({self.fetch_interval/60:.1f} min)")
        print(f"Retrain interval: {self.retrain_interval}s ({self.retrain_interval/60:.1f} min)")
        print(f"{'='*60}\n")
        
        while self.running:
            try:
                self.update_cycle()
            except Exception as e:
                print(f"Error in update cycle: {e}")
            
            # Wait until next fetch
            print(f"\nWaiting {self.fetch_interval}s until next update...")
            time.sleep(self.fetch_interval)
    
    def start_background_thread(self):
        """Start background service in separate thread."""
        thread = threading.Thread(target=self.run_background, daemon=True)
        thread.start()
        return thread
    
    def stop(self):
        """Stop the background service."""
        self.running = False


service = None


def run_service(**service_kwargs):
    """
    Run the trading service as a background worker (no HTTP server).

    Args:
        **service_kwargs: Arguments for TradingService
    """
    global service

    # Create service
    service = TradingService(**service_kwargs)
    service.initialize()

    # Start background worker
    print("Starting background worker thread...")
    thread = service.start_background_thread()

    # Keep process alive until interrupted
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping service...")
        service.stop()
        thread.join(timeout=5)


if __name__ == "__main__":
    # Run service with default settings
    run_service(
        fetch_interval=900,  # 15 minutes
        retrain_interval=3600,  # 1 hour
        sliding_window_days=30
    )
