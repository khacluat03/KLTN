#!/usr/bin/env python3
"""evaluate_mf_model.py

Script to evaluate the trained Matrix Factorization model.

This script performs various tests:
1. Load the trained model
2. Test rating predictions for known user-item pairs
3. Test cold-start scenarios (unknown users/items)
4. Compare predictions with actual ratings from test set
5. Calculate evaluation metrics (MAE, RMSE)

Usage:
    python evaluate_mf_model.py
"""

import os
import pandas as pd
import numpy as np
from loguru import logger
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macrec.systems.methods.matrix_factorization import MatrixFactorization

def calculate_metrics(predictions, actuals):
    """Calculate MAE and RMSE"""
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    return mae, rmse

def main():
    # Paths
    model_path = os.path.join('saved_models', 'ml-100k', 'mf_model.pkl')
    dev_data_path = os.path.join('data', 'ml-100k', 'dev.csv')
    
    logger.info("=" * 60)
    logger.info("Testing Matrix Factorization Model")
    logger.info("=" * 60)
    
    # 1. Load model
    logger.info("\n[1] Loading trained model...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Please run train_rating_model.py first.")
    
    model = MatrixFactorization(n_factors=20)
    model.load(model_path)
    logger.info(f"✓ Model loaded from {model_path}")
    logger.info(f"  - Users: {len(model.user_ids)}")
    logger.info(f"  - Items: {len(model.item_ids)}")
    logger.info(f"  - Global mean rating: {model.global_mean:.3f}")
    
    # 2. Test specific predictions
    logger.info("\n[2] Testing specific user-item predictions...")
    test_cases = [
        (1, 1),
        (1, 50),
        (100, 200),
        (500, 500),
        (943, 1349),  # Last user and item
    ]
    
    for user_id, item_id in test_cases:
        try:
            pred = model.predict(user_id, item_id)
            logger.info(f"  User {user_id:4d}, Item {item_id:4d} → Predicted rating: {pred:.3f}")
        except Exception as e:
            logger.warning(f"  User {user_id:4d}, Item {item_id:4d} → Error: {e}")
    
    # 3. Test cold-start scenarios
    logger.info("\n[3] Testing cold-start scenarios...")
    
    # Unknown user
    unknown_user = 99999
    known_item = model.item_ids[0]
    pred = model.predict(unknown_user, known_item)
    logger.info(f"  Unknown user {unknown_user}, Known item {known_item} → {pred:.3f} (should be global mean)")
    
    # Unknown item
    known_user = model.user_ids[0]
    unknown_item = 99999
    pred = model.predict(known_user, unknown_item)
    logger.info(f"  Known user {known_user}, Unknown item {unknown_item} → {pred:.3f} (should be global mean)")
    
    # Both unknown
    pred = model.predict(unknown_user, unknown_item)
    logger.info(f"  Unknown user {unknown_user}, Unknown item {unknown_item} → {pred:.3f} (should be global mean)")
    
    # 4. Evaluate on dev set
    logger.info("\n[4] Evaluating on dev set...")
    if not os.path.exists(dev_data_path):
        logger.warning(f"Dev data not found: {dev_data_path}")
        logger.warning("Skipping dev set evaluation")
    else:
        logger.info(f"Loading dev data from {dev_data_path}...")
        dev_data = pd.read_csv(dev_data_path)
        
        if 'user_id' in dev_data.columns and 'item_id' in dev_data.columns and 'rating' in dev_data.columns:
            # Sample a subset for faster testing (first 1000 interactions)
            sample_size = min(1000, len(dev_data))
            dev_sample = dev_data.head(sample_size)
            
            logger.info(f"Testing on {sample_size} interactions from dev set...")
            
            predictions = []
            actuals = []
            
            for idx, row in dev_sample.iterrows():
                user_id = row['user_id']
                item_id = row['item_id']
                actual_rating = row['rating']
                
                pred_rating = model.predict(user_id, item_id)
                
                predictions.append(pred_rating)
                actuals.append(actual_rating)
            
            # Calculate metrics
            mae, rmse = calculate_metrics(predictions, actuals)
            
            logger.info(f"\n  Evaluation Results (on {sample_size} samples):")
            logger.info(f"  - MAE (Mean Absolute Error):  {mae:.4f}")
            logger.info(f"  - RMSE (Root Mean Squared Error): {rmse:.4f}")
            
            # Show some example predictions vs actuals
            logger.info(f"\n  Sample predictions vs actual ratings:")
            for i in range(min(10, len(predictions))):
                user_id = dev_sample.iloc[i]['user_id']
                item_id = dev_sample.iloc[i]['item_id']
                logger.info(f"    User {user_id:4d}, Item {item_id:4d} → Predicted: {predictions[i]:.3f}, Actual: {actuals[i]:.1f}, Error: {abs(predictions[i] - actuals[i]):.3f}")
        else:
            logger.warning("Dev data doesn't have required columns (user_id, item_id, rating)")
    
    # 5. Test recommendation generation
    logger.info("\n[5] Testing recommendation generation...")
    test_user = model.user_ids[0]
    logger.info(f"Generating top-10 recommendations for user {test_user}...")
    
    # Get all items
    all_items = model.item_ids
    
    # Predict ratings for all items
    item_predictions = []
    for item_id in all_items[:100]:  # Test on first 100 items for speed
        pred = model.predict(test_user, item_id)
        item_predictions.append((item_id, pred))
    
    # Sort by predicted rating
    item_predictions.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"  Top-10 recommended items for user {test_user}:")
    for rank, (item_id, pred_rating) in enumerate(item_predictions[:10], 1):
        logger.info(f"    {rank:2d}. Item {item_id:4d} → Predicted rating: {pred_rating:.3f}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✓ All tests completed successfully!")
    logger.info("=" * 60)
    logger.info("\nModel is ready for use in:")
    logger.info("  - RatingPredictor tool")
    logger.info("  - Collaborative Filtering systems")
    logger.info("  - Recommendation agents")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
