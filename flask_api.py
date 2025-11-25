"""
Flask API Server for Bazaar Price Prediction with Minecraft Mod Integration
- Auto-trains full model on startup if not present
- Provides endpoints for Minecraft mod to fetch buy/sell recommendations
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import os
import requests
import json
import threading
import time
from datetime import datetime

from LGBMfulldata import (predict_item_three_models, train_three_model_system,
                           analyze_best_flips, analyze_best_investments, analyze_crash_watch)
from data_utils import fetch_recent_data
from mayor_utils import get_mayor_perks
import traceback


app = Flask(__name__)
CORS(app)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Global variables for THREE model artifacts
models_dict = None  # Will contain 'buy', 'sell', 'spread' models
scaler = None
feature_columns = None
mayor_data_cache = None
model_trained = False

# Cached predictions
cached_predictions = {}
predictions_file = os.path.join(SCRIPT_DIR, 'predictions_cache.json')
prediction_lock = threading.Lock()




def load_model_artifacts():
    """Load existing THREE model artifacts."""
    global models_dict, scaler, feature_columns, mayor_data_cache, model_trained
    
    print("Loading three-model system artifacts...")
    try:
        url = "https://sky.coflnet.com/api/items/bazaar/tags"
        item_ids = requests.get(url).json()

        models_dict = {}  # dictionary to store models for each item
        scaler = {}
        feature_columns = {}

        for item_id in item_ids:
            models_dict[item_id] = {
                'buy': joblib.load(os.path.join(SCRIPT_DIR, f'{item_id}_buy_lgbm_model.pkl')),
                'sell': joblib.load(os.path.join(SCRIPT_DIR, f'{item_id}_sell_lgbm_model.pkl')),
                'spread': joblib.load(os.path.join(SCRIPT_DIR, f'{item_id}_spread_lgbm_model.pkl'))
            }
            
            scaler[item_id] = joblib.load(os.path.join(SCRIPT_DIR, f'{item_id}_global_scaler.pkl'))
            feature_columns[item_id] = joblib.load(os.path.join(SCRIPT_DIR, f'{item_id}_global_feature_columns.pkl'))
        mayor_data_cache = get_mayor_perks()
        
        model_trained = True
        print("✅ Three-model system loaded successfully!")
        print("   - BUY model loaded")
        print("   - SELL model loaded")
        print("   - SPREAD model loaded")
        return True
    except Exception as e:
        print(f"❌ Error loading model artifacts: {e}")
        traceback.print_exc()
        return False


def get_available_items():
    """Get list of items that have downloaded JSON data files."""
    json_dir = "/Users/samuelbraga/Json Files"
    available_items = []
    
    try:
        if os.path.exists(json_dir):
            for filename in os.listdir(json_dir):
                item_id = filename.replace("bazaar_history_", "").replace(".pkl.gz", "")
                available_items.append(item_id)
            print(f"📁 Found {len(available_items)} items with downloaded data")
        else:
            print(f"⚠️  JSON directory not found: {json_dir}")
    except Exception as e:
        print(f"⚠️  Error scanning JSON files: {e}")
    
    return available_items


def load_cached_predictions():
    """Load predictions from cache file."""
    global cached_predictions
    try:
        if os.path.exists(predictions_file):
            with open(predictions_file, 'r') as f:
                cached_predictions = json.load(f)
            print(f"✅ Loaded {len(cached_predictions)} cached predictions")
        else:
            cached_predictions = {}
    except Exception as e:
        print(f"⚠️  Error loading cached predictions: {e}")
        cached_predictions = {}


def save_cached_predictions():
    """Save predictions to cache file."""
    try:
        with prediction_lock:
            with open(predictions_file, 'w') as f:
                json.dump(cached_predictions, f, indent=2)
    except Exception as e:
        print(f"⚠️  Error saving cached predictions: {e}")


def background_prediction_loop():
    """Background thread that continuously updates predictions for all items."""
    global cached_predictions
    
    print("\n🔄 Starting background prediction loop...")
    
    while True:
        try:
            if not model_trained:
                print("⏳ Waiting for model to be trained...")
                time.sleep(10)
                continue
            
            # Get only items that have downloaded JSON data
            available_items = get_available_items()
            if not available_items:
                print("⚠️  No items with downloaded data found, sleeping...")
                time.sleep(60)
                continue
            
            print(f"\n📊 Updating predictions for {len(available_items)} items with downloaded data...")
            
            batch_count = 0
            updated_count = 0
            
            for item_id in available_items:
                try:
                    # Make prediction with THREE models
                    prediction = predict_item_three_models(
                        models_dict[item_id], scaler[item_id], feature_columns[item_id],
                        item_id, mayor_data_cache[item_id]
                    )
                    
                    # Store comprehensive prediction in cache
                    with prediction_lock:
                        cached_predictions[item_id] = {
                            'item_id': prediction['item_id'],
                            'timestamp': prediction['timestamp'],
                            
                            # Buy price info
                            'buy_current': prediction['buy']['current_price'],
                            'buy_predicted': prediction['buy']['predicted_price'],
                            'buy_change_pct': prediction['buy']['change_pct'],
                            'buy_direction': prediction['buy']['direction'],
                            'buy_confidence': prediction['buy']['confidence'],
                            
                            # Sell price info
                            'sell_current': prediction['sell']['current_price'],
                            'sell_predicted': prediction['sell']['predicted_price'],
                            'sell_change_pct': prediction['sell']['change_pct'],
                            'sell_direction': prediction['sell']['direction'],
                            'sell_confidence': prediction['sell']['confidence'],
                            
                            # Spread info
                            'spread_current': prediction['spread']['current_spread'],
                            'spread_predicted': prediction['spread']['predicted_spread'],
                            'spread_direction': prediction['spread']['direction'],
                            'spread_confidence': prediction['spread']['confidence'],
                            
                            # Flip profit
                            'flip_profit_current': prediction['flip_profit']['current_pct'],
                            'flip_profit_predicted': prediction['flip_profit']['predicted_pct'],
                            
                            # Overall recommendation
                            'recommendation': prediction['recommendation'],
                            
                            # Legacy fields for backward compatibility
                            'action': 'BUY' if prediction['buy']['direction'] == 'UP' else 'SELL',
                            'current_price': prediction['buy']['current_price'],
                            'predicted_price': prediction['buy']['predicted_price'],
                            'expected_profit_pct': abs(prediction['buy']['change_pct']),
                            'confidence': prediction['buy']['confidence'],
                            'direction': prediction['buy']['direction']
                        }
                    
                    updated_count += 1
                    batch_count += 1
                    
                    # Save to file every 30 items
                    if batch_count >= 30:
                        save_cached_predictions()
                        print(f"  ✓ Updated {updated_count}/{len(available_items)} items (sleeping 10s)")
                        time.sleep(10)  # Sleep after every 30 items
                        batch_count = 0
                        
                except Exception as e:
                    print(f"  ✗ Error predicting {item_id}: {e}")
                    continue
            
            # Save any remaining predictions
            if batch_count > 0:
                save_cached_predictions()
            
            print(f"✅ Completed prediction cycle: {updated_count}/{len(available_items)} items updated")
            print("⏳ Sleeping 10 seconds before next cycle...")
            time.sleep(10)
            
        except Exception as e:
            print(f"❌ Error in prediction loop: {e}")
            traceback.print_exc()
            time.sleep(30)  # Wait longer on error



# Initialize on startup
@app.before_request
def ensure_model_loaded():
    """Ensure model is loaded before handling any request."""
    global model_trained
    
    if not model_trained and request.endpoint not in ['health', 'root']:
        return jsonify({
            'error': 'Model not ready',
            'message': 'Server is still initializing. Please wait.'
        }), 503


# Root endpoint
@app.route('/')
def root():
    """API information endpoint."""
    return jsonify({
        'name': 'Bazaar Price Prediction API',
        'version': '2.0.0',
        'description': 'Flask API for Minecraft Mod Integration',
        'model_ready': model_trained,
        'endpoints': {
            'health': '/health',
            'items': '/items',
            'predict': '/predict/<item_id>',
            'predict_with_data': '/predict/with-data',
            'predict_batch': '/predict/batch',
            'recommendations': '/recommendations'
        }
    })


# Health check
@app.route('/health')
def health_check():
    """Check server health and model status."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': models_dict is not None,
        'model_trained': model_trained,
        'timestamp': datetime.now().isoformat()
    })


# Single item prediction
@app.route('/predict/<item_id>')
def predict_single(item_id):
    """
    Predict price direction for a specific item using three-model system.
    """
    if models_dict is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        # Make prediction using three-model system
        prediction = predict_item_three_models(
            models_dict[item_id], scaler[item_id], feature_columns[item_id],
            item_id, mayor_data_cache
        )
        
        return jsonify(prediction)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


# Get all available items
@app.route('/items')
def get_items():
    """Get list of all available bazaar items."""
    try:
        url = "https://sky.coflnet.com/api/items/bazaar/tags"
        response = requests.get(url)
        items = response.json()
        return jsonify({
            'items': items,
            'count': len(items)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500




# Legacy endpoint - disabled as not implemented for three-model system
# @app.route('/predict/with-data', methods=['POST'])
# def predict_with_data():
#     """Disabled - not implemented for three-model system"""
#     return jsonify({'error': 'Endpoint not available in three-model system'}), 501
        
# Batch prediction
@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict prices for multiple items using recent API data.
    Body: {"item_ids": ["ITEM1", "ITEM2", ...], "hours": 24 (optional)}
    """
    if models_dict is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        item_ids = data.get('item_ids', [])
        hours = data.get('hours', 24)
        
        if not item_ids:
            return jsonify({'error': 'No item_ids provided'}), 400
        
        predictions = []
        errors = []
        
        for item_id in item_ids:
            try:
                # Make prediction using three-model system
                prediction = predict_item_three_models(
                    models_dict[item_id], scaler[item_id], feature_columns[item_id],
                    item_id, mayor_data_cache
                )
                predictions.append(prediction)
                
            except Exception as e:
                errors.append({
                    'item_id': item_id,
                    'error': str(e)
                })
        
        return jsonify({
            'predictions': predictions,
            'errors': errors,
            'total': len(item_ids),
            'successful': len(predictions),
            'failed': len(errors)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Get cached predictions (fast endpoint)
@app.route('/predictions')
def get_cached_predictions():
    """
    Get cached predictions from the background loop.
    Much faster than /recommendations as it doesn't compute predictions.
    Query params:
        - item_ids: comma-separated list of item IDs (optional)
        - limit: int (default 10)
        - min_confidence: float (default 50.0)
    """
    try:
        item_ids_param = request.args.get('item_ids', None)
        limit = int(request.args.get('limit', 100))
        min_confidence = float(request.args.get('min_confidence', 50.0))
        
        with prediction_lock:
            predictions_list = list(cached_predictions.values())
        
        # Filter by item_ids if provided
        if item_ids_param:
            requested_ids = [id.strip() for id in item_ids_param.split(',')]
            predictions_list = [p for p in predictions_list if p['item_id'] in requested_ids]
        
        # Filter by confidence
        predictions_list = [p for p in predictions_list if p['confidence'] >= min_confidence]
        
        # Sort by confidence descending
        predictions_list.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            'predictions': predictions_list[:limit],
            'total': len(predictions_list),
            'cache_size': len(cached_predictions),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Get recommendations for Minecraft mod
@app.route('/recommendations')
def get_recommendations():
    """
    Get buy/sell recommendations for Minecraft mod.
    Returns top opportunities sorted by confidence.
    Query params:
        - limit: int (default 10)
        - min_confidence: float (default 60.0)
        - hours: int (default 24) - hours of recent data to analyze
    """
    if models_dict is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        limit = int(request.args.get('limit', 100))
        min_confidence = float(request.args.get('min_confidence', 50.0))
        
        
        recommendations = []
        url = "https://sky.coflnet.com/api/items/bazaar/tags"
        all_items = requests.get(url).json()

        for item_id in all_items:
            try:
                # Make prediction using three-model system
                prediction = predict_item_three_models(
                    models_dict[item_id], scaler[item_id], feature_columns[item_id],
                    item_id, mayor_data_cache
                )
                
                # Only include confident predictions
                if prediction['confidence'] >= min_confidence:
                    recommendations.append({
                        'item_id': prediction['item_id'],
                        'action': 'BUY' if prediction['direction'] == 'UP' else 'SELL',
                        'current_price': prediction['current_price'],
                        'predicted_price': prediction['predicted_price'],
                        'expected_profit_pct': abs(prediction['predicted_change_pct']),
                        'confidence': prediction['confidence'],
                        'recommendation': prediction['recommendation']
                    })
            except Exception:
                continue
        
        # Sort by confidence descending
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            'recommendations': recommendations[:limit],
            'total_analyzed': len(all_items),
            'total_recommendations': len(recommendations),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Investment Analysis Endpoints

@app.route('/flips')
def get_best_flips():
    """
    Get best flip opportunities (largest spread where sell_order > buy_order).
    Query params:
        - limit: int (default 10)
    """
    try:
        limit = int(request.args.get('limit', 100))
        
        with prediction_lock:
            predictions_list = list(cached_predictions.values())
        
        # Analyze flips
        flips = analyze_best_flips(predictions_list, top_n=limit)
        
        return jsonify({
            'flips': flips,
            'total': len(flips),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/investments')
def get_best_investments():
    """
    Get best investment opportunities with weighted expected return.
    Query params:
        - timeframe: str (1d, 1w, 1m) - default 1d
        - limit: int (default 10)
    """
    try:
        timeframe_str = request.args.get('timeframe', '1d')
        limit = int(request.args.get('limit', 100))
        
        # Parse timeframe
        timeframe_days = {
            '1d': 1,
            '1w': 7,
            '1m': 30
        }.get(timeframe_str, 1)
        
        with prediction_lock:
            predictions_list = list(cached_predictions.values())
        
        # Analyze investments
        investments = analyze_best_investments(predictions_list, timeframe_days=timeframe_days, top_n=limit)
        
        return jsonify({
            'investments': investments,
            'timeframe': timeframe_str,
            'total': len(investments),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/crash_watch')
def get_crash_watch():
    """
    Get items with predicted BUY ORDER crashes and reversal estimates.
    Query params:
        - limit: int (default 10)
    """
    try:
        limit = int(request.args.get('limit', 100))
        
        with prediction_lock:
            predictions_list = list(cached_predictions.values())
        
        # Analyze crash watch
        crash_items = analyze_crash_watch(predictions_list, top_n=limit)
        
        return jsonify({
            'crash_items': crash_items,
            'total': len(crash_items),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    print("\n" + "="*70)
    print("BAZAAR PRICE PREDICTION FLASK API")
    print("="*70)
    
    # Check if model exists, if not train it
    if load_model_artifacts():
        print("Ready to serve predictions!")

    
    # Load cached predictions
    load_cached_predictions()
    
    # Start background prediction loop
    prediction_thread = threading.Thread(target=background_prediction_loop, daemon=True)
    prediction_thread.start()
    print("✅ Background prediction thread started")
    
    print("\n" + "="*70)
    print("🚀 Starting Flask API Server")
    print("="*70)
    print("Server running at: http://0.0.0.0:5001")
    print("Minecraft mod should connect to this endpoint")
    print("/predictions - Fast cached predictions endpoint")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False)
