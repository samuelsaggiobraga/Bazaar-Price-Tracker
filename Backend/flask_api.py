"""
Flask API Server for Bazaar Price Prediction (Entry-Based System)
- Auto-trains/predicts single-entry model per item
- Provides endpoints for Minecraft mod to fetch entry recommendations
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import os
import requests
import json
import threading
import time
from datetime import datetime, timezone
import traceback
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from LGBMfulldata import (
    predict_entries, analyze_entries
)
from Utils.mayor_utils import get_mayor_perks

app = Flask(__name__)
CORS(app)

SCRIPT_DIR = os.path.join(project_root, "Model_Files")

# Global artifacts
models_dict = {}          # {item_id: trained_model}
scalers_dict = {}         # {item_id: scaler}
feature_columns_dict = {} # {item_id: list_of_features}
mayor_data_cache = None
model_trained = False

# Cached predictions
cached_predictions = {}
predictions_file = os.path.join(SCRIPT_DIR, 'predictions_cache.json')
prediction_lock = threading.Lock()


# ------------------- Utilities -------------------

def load_model_artifacts():
    """Load existing entry-based model artifacts for all items."""
    global models_dict, scalers_dict, feature_columns_dict, mayor_data_cache, model_trained
    try:
        url = "https://sky.coflnet.com/api/items/bazaar/tags"
        item_ids = requests.get(url).json()

        mayor_data_cache = get_mayor_perks()
        models_dict.clear()
        scalers_dict.clear()
        feature_columns_dict.clear()

        for item_id in item_ids:
            model_path = os.path.join(SCRIPT_DIR, f'{item_id}_entry_model.pkl')
            scaler_path = os.path.join(SCRIPT_DIR, f'{item_id}_entry_scaler.pkl')
            features_path = os.path.join(SCRIPT_DIR, f'{item_id}_entry_features.pkl')

            if os.path.exists(model_path):
                models_dict[item_id] = {"model_path": model_path}
                scalers_dict[item_id] = {"scaler_path": scaler_path}
                feature_columns_dict[item_id] = {"features_path": features_path}

        # Only mark the model as ready if we actually found at least one
        # trained item. The previous check against an empty dict view was
        # always ``True``.
        model_trained = len(models_dict) > 0
        print(f"✅ Loaded entry-based models: {len(models_dict)} items")
        return model_trained

    except Exception as e:
        print(f"❌ Error loading model artifacts: {e}")
        traceback.print_exc()
        return False


def get_available_items():
    """Return item IDs with downloaded JSON data."""
    json_dir = SCRIPT_DIR
    items = []
    try:
        if os.path.exists(json_dir):
            for fname in os.listdir(json_dir):
                if not fname.endswith("_entry_model.pkl"):
                    continue
                item_id = fname.replace("_entry_model.pkl", "")
                items.append(item_id)
        return items
    except Exception as e:
        print(f"⚠️ Error scanning JSON files: {e}")
        return []


def load_cached_predictions():
    """Load predictions from file."""
    global cached_predictions
    try:
        if os.path.exists(predictions_file):
            with open(predictions_file, 'r') as f:
                cached_predictions = json.load(f)
        else:
            cached_predictions = {}
    except Exception as e:
        print(f"⚠️ Error loading cached predictions: {e}")
        cached_predictions = {}


def save_cached_predictions():
    """Save predictions to file."""
    try:
        with prediction_lock:
            with open(predictions_file, 'w') as f:
                json.dump(cached_predictions, f, indent=2)
    except Exception as e:
        print(f"⚠️ Error saving cached predictions: {e}")


# ------------------- Background Prediction Loop -------------------

def background_prediction_loop():
    """Continuously update cached entry predictions."""
    global cached_predictions
    print("🔄 Starting background prediction loop...")

    while True:
        try:
            if not model_trained:
                print("⏳ Waiting for models to load...")
                time.sleep(10)
                continue

            items = get_available_items()
            if not items:
                time.sleep(60)
                continue

            for item_id in items:
                try:
                    model = joblib.load(models_dict[item_id]["model_path"])
                    scaler = joblib.load(scalers_dict[item_id]["scaler_path"])
                    features = joblib.load(feature_columns_dict[item_id]["features_path"])
                    mayor_data = mayor_data_cache

                    if not model or not scaler or not features:
                        continue

                    prediction_df = predict_entries(model, scaler, features, item_id, mayor_data)
                    ranked_entries = analyze_entries(prediction_df)

                    with prediction_lock:
                        cached_predictions[item_id] = {
                            'item_id': item_id,
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'entries': ranked_entries
                        }

                except Exception as e:
                    print(f"✗ Error predicting {item_id}: {e}")
                    continue

            save_cached_predictions()
            print(f"✅ Prediction cycle complete: {len(items)} items updated")
            time.sleep(10)

        except Exception as e:
            print(f"❌ Prediction loop error: {e}")
            traceback.print_exc()
            time.sleep(30)


# ------------------- Flask Endpoints -------------------

@app.before_request
def ensure_model_loaded():
    """Return 503 if models not loaded."""
    global model_trained
    if not model_trained and request.endpoint not in ['health', 'root']:
        return jsonify({'error': 'Model not ready', 'message': 'Server initializing'}), 503


@app.route('/')
def root():
    return jsonify({
        'name': 'Bazaar Entry Prediction API',
        'version': '1.0.0',
        'model_ready': model_trained,
        'endpoints': {
            'health': '/health',
            'items': '/items',
            'predict': '/predict/<item_id>',
            'predictions': '/predictions',
            'investments': '/investments'
        }
    })


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_trained': model_trained,
        'timestamp': datetime.now(timezone.utc).isoformat()
    })


@app.route('/items')
def get_items():
    try:
        url = "https://sky.coflnet.com/api/items/bazaar/tags"
        items = requests.get(url).json()
        return jsonify({'items': items, 'count': len(items)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/<item_id>')
def predict_single(item_id):
    try:
        model = joblib.load(models_dict[item_id]["model_path"])
        scaler = joblib.load(scalers_dict[item_id]["scaler_path"])
        features = joblib.load(feature_columns_dict[item_id]["features_path"])
        mayor_data = mayor_data_cache

        if not model or not scaler or not features:
            return jsonify({'error': 'Model not found for this item'}), 404

        df_pred = predict_entries(model, scaler, features, item_id, mayor_data)
        ranked_entries = analyze_entries(df_pred)

        return jsonify({
            'item_id': item_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'entries': ranked_entries
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predictions')
def get_cached():
    """Return best upcoming positive entry per item for homescreen ranking.

    For each item in ``cached_predictions`` we:
    - assume ``entries`` has already been processed by ``analyze_entries``
      to keep only positive, time-annotated signals
    - take the first entry (closest-in-time positive signal)
    - build a flat summary row used by the Minecraft GUI homescreen

    Items are ranked primarily by how soon the positive entry occurs
    (smallest ``delta_minutes`` first), then by ``entry_score``.
    """
    try:
        limit = int(request.args.get('limit', 100))
        min_score = float(request.args.get('min_score', 0.0))

        with prediction_lock:
            preds = list(cached_predictions.values())

        ranked_items = []
        now = datetime.now(timezone.utc)

        for p in preds:
            item_id = p.get('item_id')
            entries = p.get('entries') or []
            if not item_id or not entries:
                continue

            # Entries from analyze_entries are already positive and sorted by
            # delta_minutes ascending; take the closest one.
            best = entries[0]
            score = float(best.get('entry_score', 0.0))
            if score <= min_score:
                continue

            # Ensure we have a usable delta_minutes; if missing, recompute
            # defensively from the timestamp.
            delta_minutes = best.get('delta_minutes')
            if delta_minutes is None:
                ts_str = best.get('timestamp')
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        delta_minutes = (ts - now).total_seconds() / 60.0
                    except Exception:
                        continue
                else:
                    continue

            ranked_items.append({
                'item_id': item_id,
                'timestamp': best.get('timestamp'),
                'buy_price': best.get('buy_price'),
                'sell_price': best.get('sell_price'),
                'entry_score': score,
                'delta_minutes': float(delta_minutes),
            })

        # Sort by time-to-entry, then by score
        ranked_items.sort(key=lambda x: (x['delta_minutes'], -x['entry_score']))

        return jsonify({'items': ranked_items[:limit], 'total': len(ranked_items)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/investments')
def best_investments():
    try:
        limit = int(request.args.get('limit', 10))
        with prediction_lock:
            preds = list(cached_predictions.values())

        investments = analyze_entries(preds, top_n=limit)
        return jsonify({
            'investments': investments,
            'total': len(investments),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ------------------- Startup -------------------

if __name__ == '__main__':
    print("🚀 Starting Entry-Based Bazaar Prediction Server")

    if load_model_artifacts():
        print("✅ Models loaded, ready to serve predictions")
    else:
        print("⚠️ No models found, please train first using train_model_system()")

    load_cached_predictions()

    t = threading.Thread(target=background_prediction_loop, daemon=True)
    t.start()
    print("✅ Background prediction thread started")

    app.run(host='0.0.0.0', port=5001, debug=False)
