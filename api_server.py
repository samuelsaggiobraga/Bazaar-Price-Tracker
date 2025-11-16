"""
FastAPI Backend Server for Bazaar Price Prediction
Provides REST API endpoints for price predictions and incremental model updates.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import joblib
import os
from datetime import datetime

from LGBMfulldata import predict_item, train_global_model
from incremental_learning import IncrementalLearner
from data_utils import load_or_fetch_item_data, fetch_all_data
from mayor_utils import get_mayor_perks
import requests


# Initialize FastAPI app
app = FastAPI(
    title="Bazaar Price Prediction API",
    description="REST API for predicting bazaar item prices and incremental model updates",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global variables for model artifacts
model = None
scaler = None
feature_columns = None
item_encoder = None
incremental_learner = None
mayor_data_cache = None


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    item_id: str
    fetch_latest: bool = False


class PredictionResponse(BaseModel):
    item_id: str
    current_price: float
    predicted_price: float
    predicted_change_pct: float
    direction: str
    confidence: float
    raw_probability: float
    timestamp: str
    recommendation: str


class UpdateRequest(BaseModel):
    item_ids: List[str]
    num_boost_round: int = 50


class UpdateResponse(BaseModel):
    status: str
    message: str
    results: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    incremental_learner_available: bool
    timestamp: str


# Startup event to load model
@app.on_event("startup")
async def startup_event():
    """Load model artifacts on server startup."""
    global model, scaler, feature_columns, item_encoder, incremental_learner, mayor_data_cache
    
    print("Loading model artifacts...")
    
    # Check if model files exist
    model_files = [
        'global_lgbm_model.pkl',
        'global_scaler.pkl',
        'global_feature_columns.pkl',
        'item_encoder.pkl'
    ]
    
    if all(os.path.exists(f) for f in model_files):
        try:
            model = joblib.load('global_lgbm_model.pkl')
            scaler = joblib.load('global_scaler.pkl')
            feature_columns = joblib.load('global_feature_columns.pkl')
            item_encoder = joblib.load('item_encoder.pkl')
            
            # Initialize incremental learner
            incremental_learner = IncrementalLearner()
            
            # Cache mayor data
            mayor_data_cache = get_mayor_perks()
            
            print("Model artifacts loaded successfully!")
        except Exception as e:
            print(f"Error loading model artifacts: {e}")
            print("Server will start but predictions will fail until model is trained.")
    else:
        print("Model artifacts not found. Train a model first.")
        print("Server will start but predictions will fail until model is trained.")


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and model status."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "incremental_learner_available": incremental_learner is not None,
        "timestamp": datetime.now().isoformat()
    }


# Get all available item IDs
@app.get("/items")
async def get_items():
    """Get list of all available bazaar items."""
    try:
        url = "https://sky.coflnet.com/api/items/bazaar/tags"
        response = requests.get(url)
        items = response.json()
        return {"items": items, "count": len(items)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching items: {str(e)}")


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict price direction for a specific item.
    
    Args:
        request: PredictionRequest containing item_id and fetch_latest flag
    
    Returns:
        PredictionResponse with prediction details
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first."
        )
    
    try:
        # Load or fetch item data
        data = load_or_fetch_item_data(
            request.item_id,
            fetch_if_missing=request.fetch_latest
        )
        
        if data is None or len(data) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No data available for item {request.item_id}"
            )
        
        # Make prediction
        prediction = predict_item(
            model,
            scaler,
            feature_columns,
            item_encoder,
            request.item_id,
            data,
            mayor_data_cache
        )
        
        return prediction
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(item_ids: List[str]):
    """
    Predict prices for multiple items.
    
    Args:
        item_ids: List of item IDs to predict
    
    Returns:
        List of predictions
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first."
        )
    
    predictions = []
    errors = []
    
    for item_id in item_ids:
        try:
            data = load_or_fetch_item_data(item_id, fetch_if_missing=False)
            
            if data is None:
                errors.append({
                    "item_id": item_id,
                    "error": "No data available"
                })
                continue
            
            prediction = predict_item(
                model,
                scaler,
                feature_columns,
                item_encoder,
                item_id,
                data,
                mayor_data_cache
            )
            predictions.append(prediction)
            
        except Exception as e:
            errors.append({
                "item_id": item_id,
                "error": str(e)
            })
    
    return {
        "predictions": predictions,
        "errors": errors,
        "total": len(item_ids),
        "successful": len(predictions),
        "failed": len(errors)
    }


# Incremental update endpoint
@app.post("/update", response_model=UpdateResponse)
async def update_model(request: UpdateRequest, background_tasks: BackgroundTasks):
    """
    Incrementally update the model with new data for specified items.
    
    Args:
        request: UpdateRequest containing item_ids and num_boost_round
        background_tasks: FastAPI background tasks
    
    Returns:
        UpdateResponse with update results
    """
    if incremental_learner is None:
        raise HTTPException(
            status_code=503,
            detail="Incremental learner not available. Train a model first."
        )
    
    try:
        # Run update in background
        def run_update():
            return incremental_learner.update_multiple_items(
                request.item_ids,
                num_boost_round=request.num_boost_round
            )
        
        # For now, run synchronously (can be moved to background for async)
        results = run_update()
        
        # Reload model artifacts after update
        global model, scaler, feature_columns, item_encoder
        model = incremental_learner.model
        scaler = incremental_learner.scaler
        feature_columns = incremental_learner.feature_columns
        item_encoder = incremental_learner.item_encoder
        
        success_count = sum(1 for r in results if r.get('status') == 'success')
        
        return {
            "status": "completed",
            "message": f"Updated model with {success_count}/{len(request.item_ids)} items",
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update error: {str(e)}")


# Get update history
@app.get("/update/history")
async def get_update_history():
    """Get the history of incremental model updates."""
    if incremental_learner is None:
        raise HTTPException(
            status_code=503,
            detail="Incremental learner not available."
        )
    
    return incremental_learner.get_update_history()


# Train new model endpoint (admin only in production)
@app.post("/train")
async def train_model(item_ids: Optional[List[str]] = None, limit: int = 10):
    """
    Train a new global model from scratch.
    
    Args:
        item_ids: Optional list of specific item IDs to train on
        limit: Maximum number of items to train on (default 10)
    
    Returns:
        Training status
    """
    try:
        if item_ids is None:
            # Fetch all items
            url = "https://sky.coflnet.com/api/items/bazaar/tags"
            all_items = requests.get(url).json()
            item_ids = all_items[:limit]
        
        # Train model
        global model, scaler, feature_columns, item_encoder, incremental_learner
        
        model, scaler, feature_columns, item_encoder = train_global_model(item_ids)
        
        # Reinitialize incremental learner
        incremental_learner = IncrementalLearner()
        
        return {
            "status": "success",
            "message": f"Model trained on {len(item_ids)} items",
            "item_ids": item_ids
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")


# Get model info
@app.get("/model/info")
async def get_model_info():
    """Get information about the current model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "LightGBM Binary Classifier",
        "num_features": len(feature_columns) if feature_columns else 0,
        "features": feature_columns if feature_columns else [],
        "num_items_encoded": len(item_encoder.classes_) if item_encoder else 0,
        "model_loaded": True
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Bazaar Price Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "items": "/items",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "update": "/update",
            "history": "/update/history",
            "train": "/train",
            "model_info": "/model/info"
        }
    }


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
