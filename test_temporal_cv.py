#!/usr/bin/env python3
"""
Test script for the temporal CV model (two-phase training with pre/post mayor data splits).
This validates the core ML model used in production.
"""

import json
import joblib
import os
from LGBMfulldata import predict_item
from data_utils import load_or_fetch_item_data
from mayor_utils import get_mayor_perks


def test_temporal_cv_model():
    """Test the temporal CV model with pre/post mayor data splits."""
    print("=" * 70)
    print("TEMPORAL CV MODEL TEST")
    print("=" * 70)
    
    # Check if model exists
    model_files = [
        'global_lgbm_model.pkl',
        'global_scaler.pkl',
        'global_feature_columns.pkl',
        'item_encoder.pkl'
    ]
    
    if not all(os.path.exists(f) for f in model_files):
        print("❌ Model files not found.")
        print("Run: python train_with_cv.py --mode two_phase")
        return False
    
    # Load model
    print("\n✓ Loading model artifacts...")
    model = joblib.load('global_lgbm_model.pkl')
    scaler = joblib.load('global_scaler.pkl')
    feature_columns = joblib.load('global_feature_columns.pkl')
    item_encoder = joblib.load('item_encoder.pkl')
    
    # Get test items
    test_items = list(item_encoder.classes_)[:3]  # Test with 3 items
    print(f"✓ Testing with {len(test_items)} items")
    
    # Get mayor data
    print("✓ Loading mayor data...")
    mayor_data = get_mayor_perks()
    
    # Test predictions for each item
    successful_predictions = 0
    
    for test_item in test_items:
        print(f"\n{'─' * 70}")
        print(f"Testing item: {test_item}")
        print(f"{'─' * 70}")
        
        # Load data
        data = load_or_fetch_item_data(test_item, fetch_if_missing=False)
        
        if data is None or len(data) == 0:
            print(f"⚠️  No data available for {test_item}, skipping...")
            continue
        
        # Make prediction
        try:
            prediction = predict_item(
                model, scaler, feature_columns, item_encoder,
                test_item, data, mayor_data
            )
            
            # Verify prediction format
            required_fields = [
                'item_id', 'current_price', 'predicted_price',
                'predicted_change_pct', 'direction', 'confidence',
                'recommendation'
            ]
            
            if not all(field in prediction for field in required_fields):
                print(f"❌ Prediction missing required fields")
                continue
            
            # Display results
            print(f"📊 Item: {prediction['item_id']}")
            print(f"💰 Current Price: ${prediction['current_price']:,.2f}")
            print(f"🔮 Predicted Price: ${prediction['predicted_price']:,.2f}")
            print(f"📈 Expected Change: {prediction['predicted_change_pct']:+.2f}%")
            print(f"🎯 Direction: {prediction['direction']}")
            print(f"✨ Confidence: {prediction['confidence']:.1f}%")
            print(f"💡 Recommendation: {prediction['recommendation']}")
            
            # Validate values
            if prediction['current_price'] > 0 and prediction['confidence'] > 0:
                print("✅ Prediction valid")
                successful_predictions += 1
            else:
                print("⚠️  Warning: Unusual prediction values")
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("TEST SUMMARY")
    print(f"{'=' * 70}")
    print(f"Successful predictions: {successful_predictions}/{len(test_items)}")
    
    if successful_predictions > 0:
        print("\n✅ Temporal CV model is working correctly!")
        return True
    else:
        print("\n❌ No successful predictions. Check model training.")
        return False


if __name__ == '__main__':
    success = test_temporal_cv_model()
    exit(0 if success else 1)
