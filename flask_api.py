#!/usr/bin/env python3
"""
Flask API Server for Bazaar Price Prediction with Minecraft Mod Integration
- Auto-trains full model on startup if not present
- Supports incremental learning with 90/10 split and mayor data
- Provides endpoints for Minecraft mod to fetch buy/sell recommendations
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import os
import requests
from datetime import datetime

from LGBMfulldata import predict_item, train_two_phase_model, train_global_model
from incremental_learning import IncrementalLearner
from data_utils import load_or_fetch_item_data, fetch_recent_data
from mayor_utils import get_mayor_perks


app = Flask(__name__)
CORS(app)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Global variables for model artifacts
model = None
scaler = None
feature_columns = None
item_encoder = None
incremental_learner = None
mayor_data_cache = None
model_trained = False


def check_model_exists():
    """Check if all required model files exist."""
    model_files = [
        'global_lgbm_model.pkl',
        'global_scaler.pkl',
        'global_feature_columns.pkl',
        'item_encoder.pkl'
    ]
    return all(os.path.exists(os.path.join(SCRIPT_DIR, f)) for f in model_files)


def load_model_artifacts():
    """Load existing model artifacts."""
    global model, scaler, feature_columns, item_encoder, incremental_learner, mayor_data_cache, model_trained
    
    print("Loading model artifacts...")
    try:
        model = joblib.load(os.path.join(SCRIPT_DIR, 'global_lgbm_model.pkl'))
        scaler = joblib.load(os.path.join(SCRIPT_DIR, 'global_scaler.pkl'))
        feature_columns = joblib.load(os.path.join(SCRIPT_DIR, 'global_feature_columns.pkl'))
        item_encoder = joblib.load(os.path.join(SCRIPT_DIR, 'item_encoder.pkl'))
        
        # Initialize incremental learner
        incremental_learner = IncrementalLearner()
        
        # Cache mayor data
        mayor_data_cache = get_mayor_perks()
        
        model_trained = True
        print("✅ Model artifacts loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model artifacts: {e}")
        return False


def train_full_model():
    """Train the full model using two-phase temporal CV."""
    global model, scaler, feature_columns, item_encoder, incremental_learner, mayor_data_cache, model_trained
    
    print("\n" + "="*70)
    print("MODEL NOT FOUND - TRAINING FULL MODEL")
    print("="*70)
    print("This may take several minutes...")
    
    try:
        # Fetch all item IDs
        print("\nFetching item IDs from API...")
        url = "https://sky.coflnet.com/api/items/bazaar/tags"
        item_ids = requests.get(url).json()
        print(f"Training on {len(item_ids)} items with two-phase temporal CV")
        
        # Train using two-phase approach (pre/post mayor data splits)
        model, scaler, feature_columns, item_encoder = train_two_phase_model(item_ids)
        
        if model is None:
            print("❌ Training failed - no valid data found.")
            return False
        
        # Initialize incremental learner
        incremental_learner = IncrementalLearner()
        
        # Cache mayor data
        mayor_data_cache = get_mayor_perks()
        
        model_trained = True
        
        print("\n" + "="*70)
        print("✅ MODEL TRAINING COMPLETE")
        print("="*70)
        print("Model artifacts saved and ready for predictions!")
        
        return True
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False


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
            'predict_batch': '/predict/batch',
            'recommendations': '/recommendations',
            'update_incremental': '/update/incremental',
            'update_history': '/update/history'
        }
    })


# Health check
@app.route('/health')
def health_check():
    """Check server health and model status."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_trained': model_trained,
        'incremental_learner_available': incremental_learner is not None,
        'timestamp': datetime.now().isoformat()
    })


# Single item prediction
@app.route('/predict/<item_id>')
def predict_single(item_id):
    """
    Predict price direction for a specific item using recent API data.
    Query params:
        - hours: int (optional, default 24) - hours of recent data to fetch
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        hours = int(request.args.get('hours', 24))
        
        # Fetch only recent data from API
        print(f"Fetching last {hours} hours of data for {item_id}...")
        data = fetch_recent_data(item_id, hours=hours)
        
        if data is None or len(data) == 0:
            return jsonify({
                'error': f'No recent data available for item {item_id}'
            }), 404
        
        # Make prediction using recent data only
        prediction = predict_item(
            model, scaler, feature_columns, item_encoder,
            item_id, data, mayor_data_cache
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




# Batch prediction
@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict prices for multiple items using recent API data.
    Body: {"item_ids": ["ITEM1", "ITEM2", ...], "hours": 24 (optional)}
    """
    if model is None:
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
                # Fetch recent data from API
                item_data = fetch_recent_data(item_id, hours=hours)
                
                if item_data is None or len(item_data) == 0:
                    errors.append({
                        'item_id': item_id,
                        'error': 'No recent data available'
                    })
                    continue
                
                prediction = predict_item(
                    model, scaler, feature_columns, item_encoder,
                    item_id, item_data, mayor_data_cache
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
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        limit = int(request.args.get('limit', 10))
        min_confidence = float(request.args.get('min_confidence', 50.0))
        hours = int(request.args.get('hours', 168))
        
        # Get all items from encoder
        all_items = list(item_encoder.classes_)  # Use all items the model was trained on
        
        recommendations = []
        
        for item_id in all_items:
            try:
                # Fetch recent data from API
                item_data = fetch_recent_data(item_id, hours=hours)
                if item_data is None or len(item_data) == 0:
                    continue
                
                prediction = predict_item(
                    model, scaler, feature_columns, item_encoder,
                    item_id, item_data, mayor_data_cache
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


# Incremental learning with 90/10 split
@app.route('/update/incremental', methods=['POST'])
def update_incremental():
    """
    Incrementally update model with recent data (90/10 split with mayor data).
    Body: {
        "item_ids": ["ITEM1", "ITEM2", ...],
        "num_boost_round": 50 (optional)
    }
    """
    if incremental_learner is None:
        return jsonify({'error': 'Incremental learner not available'}), 503
    
    try:
        data = request.get_json()
        item_ids = data.get('item_ids', [])
        num_boost_round = data.get('num_boost_round', 50)
        
        if not item_ids:
            return jsonify({'error': 'No item_ids provided'}), 400
        
        # Perform incremental update
        results = incremental_learner.update_multiple_items(
            item_ids,
            num_boost_round=num_boost_round
        )
        
        # Reload model artifacts
        global model, scaler, feature_columns, item_encoder
        model = incremental_learner.model
        scaler = incremental_learner.scaler
        feature_columns = incremental_learner.feature_columns
        item_encoder = incremental_learner.item_encoder
        
        success_count = sum(1 for r in results if r.get('status') == 'success')
        
        return jsonify({
            'status': 'completed',
            'message': f'Updated model with {success_count}/{len(item_ids)} items',
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Get update history
@app.route('/update/history')
def get_update_history():
    """Get the history of incremental model updates."""
    if incremental_learner is None:
        return jsonify({'error': 'Incremental learner not available'}), 503
    
    return jsonify(incremental_learner.get_update_history())


if __name__ == '__main__':
    print("\n" + "="*70)
    print("BAZAAR PRICE PREDICTION FLASK API")
    print("="*70)
    
    # Check if model exists, if not train it
    if check_model_exists():
        print("\n✅ Model files found - loading existing model...")
        if load_model_artifacts():
            print("Ready to serve predictions!")
        else:
            print("Failed to load model, attempting to train...")
            if not train_full_model():
                print("❌ Cannot start server without a trained model.")
                exit(1)
    else:
        print("\n⚠️  Model files not found - training full model...")
        if not train_full_model():
            print("❌ Cannot start server without a trained model.")
            exit(1)
    
    print("\n" + "="*70)
    print("🚀 Starting Flask API Server")
    print("="*70)
    print("Server running at: http://0.0.0.0:5001")
    print("Minecraft mod should connect to this endpoint")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False)
