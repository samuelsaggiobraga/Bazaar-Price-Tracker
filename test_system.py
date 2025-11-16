"""
Test script for the complete Bazaar Price Prediction system.
Tests prediction improvements and incremental learning.
"""

import json
import joblib
import os
from LGBMfulldata import predict_item
from data_utils import load_or_fetch_item_data
from mayor_utils import get_mayor_perks


def test_prediction_output():
    """Test that predictions return meaningful, user-friendly output."""
    print("=" * 70)
    print("TEST 1: Prediction Output Format")
    print("=" * 70)
    
    # Check if model exists
    model_files = [
        'global_lgbm_model.pkl',
        'global_scaler.pkl',
        'global_feature_columns.pkl',
        'item_encoder.pkl'
    ]
    
    if not all(os.path.exists(f) for f in model_files):
        print("❌ Model files not found. Run LGBMfulldata.py first to train a model.")
        return False
    
    # Load model
    print("\n✓ Loading model artifacts...")
    model = joblib.load('global_lgbm_model.pkl')
    scaler = joblib.load('global_scaler.pkl')
    feature_columns = joblib.load('global_feature_columns.pkl')
    item_encoder = joblib.load('item_encoder.pkl')
    
    # Get test item
    test_items = list(item_encoder.classes_)
    if not test_items:
        print("❌ No items found in encoder")
        return False
    
    test_item = test_items[0]
    print(f"✓ Testing with item: {test_item}")
    
    # Load data
    print("✓ Loading item data...")
    data = load_or_fetch_item_data(test_item, fetch_if_missing=False)
    
    if data is None:
        print(f"❌ No data available for {test_item}")
        return False
    
    # Get mayor data
    print("✓ Loading mayor data...")
    mayor_data = get_mayor_perks()
    
    # Make prediction
    print("✓ Making prediction...")
    try:
        prediction = predict_item(model, scaler, feature_columns, item_encoder, 
                                 test_item, data, mayor_data)
        
        # Verify prediction format
        print("\n" + "=" * 70)
        print("PREDICTION RESULT")
        print("=" * 70)
        
        required_fields = [
            'item_id', 'current_price', 'predicted_price', 
            'predicted_change_pct', 'direction', 'confidence',
            'recommendation'
        ]
        
        all_fields_present = all(field in prediction for field in required_fields)
        
        if not all_fields_present:
            print("❌ Prediction missing required fields")
            return False
        
        # Pretty print results
        print(f"\n📊 Item: {prediction['item_id']}")
        print(f"💰 Current Price: ${prediction['current_price']:,.2f}")
        print(f"🔮 Predicted Price: ${prediction['predicted_price']:,.2f}")
        print(f"📈 Expected Change: {prediction['predicted_change_pct']:+.2f}%")
        print(f"🎯 Direction: {prediction['direction']}")
        print(f"✨ Confidence: {prediction['confidence']:.1f}%")
        print(f"💡 Recommendation: {prediction['recommendation']}")
        print(f"⏰ Timestamp: {prediction['timestamp']}")
        
        # Verify values are not all zeros
        if prediction['current_price'] == 0:
            print("\n⚠️  WARNING: Current price is zero - may indicate data issue")
        elif prediction['confidence'] == 0:
            print("\n⚠️  WARNING: Confidence is zero - may indicate model issue")
        else:
            print("\n✅ Prediction format looks good!")
        
        print("\n" + "=" * 70)
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_incremental_learning():
    """Test incremental learning functionality."""
    print("\n\n" + "=" * 70)
    print("TEST 2: Incremental Learning")
    print("=" * 70)
    
    try:
        from incremental_learning import IncrementalLearner
        
        print("\n✓ Loading incremental learner...")
        learner = IncrementalLearner()
        
        # Get test item
        test_items = list(learner.item_encoder.classes_)
        test_item = test_items[0]
        
        print(f"✓ Testing with item: {test_item}")
        
        # Load data
        print("✓ Loading item data...")
        data = load_or_fetch_item_data(test_item, fetch_if_missing=False)
        
        if data is None:
            print(f"❌ No data available for {test_item}")
            return False
        
        # Use last 500 entries for update
        update_data = data[-500:]
        print(f"✓ Using {len(update_data)} data points for update")
        
        # Perform incremental update
        print("✓ Performing incremental update (this may take a moment)...")
        result = learner.update_with_new_data(
            test_item,
            update_data,
            num_boost_round=10  # Small number for testing
        )
        
        print("\n" + "=" * 70)
        print("UPDATE RESULT")
        print("=" * 70)
        print(json.dumps(result, indent=2))
        
        if result['status'] == 'success':
            print("\n✅ Incremental learning works!")
            
            # Show update history
            history = learner.get_update_history()
            print("\n" + "=" * 70)
            print("UPDATE HISTORY")
            print("=" * 70)
            print(f"Total updates: {history['total_updates']}")
            print(f"Last update: {history['last_update']}")
            
            return True
        else:
            print("\n❌ Update failed")
            return False
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Train a model first using LGBMfulldata.py")
        return False
    except Exception as e:
        print(f"❌ Incremental learning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_availability():
    """Test if API dependencies are available."""
    print("\n\n" + "=" * 70)
    print("TEST 3: API Dependencies")
    print("=" * 70)
    
    try:
        import fastapi
        import uvicorn
        import pydantic
        print("✅ FastAPI and dependencies available")
        print(f"   - FastAPI version: {fastapi.__version__}")
        print(f"   - Uvicorn version: {uvicorn.__version__}")
        print(f"   - Pydantic version: {pydantic.__version__}")
        print("\n💡 You can now run the API server with:")
        print("   python api_server.py")
        return True
    except ImportError as e:
        print(f"⚠️  API dependencies not installed: {e}")
        print("\n💡 Install with: pip install -r requirements.txt")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("BAZAAR PRICE PREDICTION SYSTEM TEST SUITE")
    print("=" * 70)
    
    results = {
        'Prediction Output': test_prediction_output(),
        'Incremental Learning': test_incremental_learning(),
        'API Dependencies': test_api_availability()
    }
    
    # Summary
    print("\n\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! System is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
