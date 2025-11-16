#!/usr/bin/env python3
"""
Training script with cross-validation for LightGBM model.

This script provides two training modes:
1. Two-phase temporal: Automatically splits at mayor data availability
   - Phase 1: Pre-mayor data (3-fold CV, 80/20 split)
   - Phase 2: Post-mayor data (5-fold CV, 70/30 split)
2. Manual mode: Choose specific configuration

Usage:
    python train_with_cv.py --mode two_phase   # Recommended: automatic temporal split
    python train_with_cv.py --mode no_mayor    # Manual: train without mayor data (5-fold, 80/20)
    python train_with_cv.py --mode with_mayor  # Manual: train with mayor data (5-fold, 70/30)
"""

import argparse
import requests
from LGBMfulldata import train_global_model, train_two_phase_model
from data_utils import load_or_fetch_item_data
from mayor_utils import get_mayor_perks


def main():
    parser = argparse.ArgumentParser(
        description='Train LightGBM model with cross-validation'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['two_phase', 'no_mayor', 'with_mayor'],
        default='two_phase',
        help='Training mode: two_phase (recommended), no_mayor (5-fold 80/20), or with_mayor (5-fold 70/30)'
    )
    parser.add_argument(
        '--n-items',
        type=int,
        default=None,
        help='Number of items to train on (default: all items)'
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of folds for cross-validation (default: 5, only used in manual modes)'
    )
    
    args = parser.parse_args()
    
    # Fetch all item IDs
    print("Fetching item IDs from API...")
    url = "https://sky.coflnet.com/api/items/bazaar/tags"
    item_ids = requests.get(url).json()
    
    if args.n_items:
        item_ids = item_ids[:args.n_items]
        print(f"Training on {args.n_items} items")
    else:
        print(f"Training on all {len(item_ids)} items")
    
    # Configure training based on mode
    if args.mode == 'two_phase':
        # Recommended: automatic temporal split
        model, scaler, feature_columns, item_encoder = train_two_phase_model(item_ids)
    
    elif args.mode == 'no_mayor':
        print("\n" + "="*60)
        print("MODE: Training WITHOUT Mayor Data (Manual)")
        print(f"Configuration: {args.n_folds}-fold CV with 80/20 train/val split")
        print("="*60 + "\n")
        
        model, scaler, feature_columns, item_encoder = train_global_model(
            item_ids,
            use_mayor_data=False,
            n_folds=args.n_folds,
            train_val_split=0.8
        )
        
    else:  # with_mayor
        print("\n" + "="*60)
        print("MODE: Training WITH Mayor Data (Manual)")
        print(f"Configuration: {args.n_folds}-fold CV with 70/30 train/val split")
        print("="*60 + "\n")
        
        model, scaler, feature_columns, item_encoder = train_global_model(
            item_ids,
            use_mayor_data=True,
            n_folds=args.n_folds,
            train_val_split=0.7
        )
    
    if model is None:
        print("\nTraining failed - no valid data found.")
        return 1
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nModel artifacts saved:")
    print("  - global_lgbm_model.pkl")
    print("  - global_scaler.pkl")
    print("  - global_feature_columns.pkl")
    print("  - item_encoder.pkl")
    print("  - model_metrics.json")
    print("\nYou can now use the trained model for predictions.")
    
    return 0


if __name__ == '__main__':
    exit(main())
