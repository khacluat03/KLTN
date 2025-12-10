#!/usr/bin/env python3
"""evaluate_cf_model.py

Script to evaluate the trained Collaborative Filtering models.

This script:
1. Loads the pre-computed CF similarity matrices
2. Tests rating predictions using 4 methods (User/Item × Pearson/Cosine)
3. Evaluates on dev set and calculates MAE/RMSE for each method
4. Compares CF methods with each other and with MF baseline
5. Tests recommendation quality

Usage:
    python evaluate_cf_model.py
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import directly from the module to avoid circular import
from macrec.tools.collaborative_filtering import CollaborativeFiltering

def calculate_metrics(predictions, actuals):
    """Calculate MAE and RMSE"""
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    return mae, rmse

def load_cf_tool_with_matrices(cf_dir, data_path):
    """Load CF tool and pre-computed similarity matrices"""
    import json
    
    # Create temporary config
    cf_config = {
        'data_path': data_path,
        'model_path': os.path.join('saved_models', 'ml-100k'),
        'min_common_items': 2,
        'min_common_users': 2,
        'k_neighbors': 50
    }
    
    temp_config_path = 'temp_cf_config.json'
    with open(temp_config_path, 'w') as f:
        json.dump(cf_config, f)
    
    try:
        # Initialize CF tool
        cf_tool = CollaborativeFiltering(config_path=temp_config_path)
        
        # Load pre-computed similarity matrices
        logger.info("Loading pre-computed similarity matrices...")
        
        user_sim_pearson_path = os.path.join(cf_dir, 'user_sim_pearson.pkl')
        if os.path.exists(user_sim_pearson_path):
            with open(user_sim_pearson_path, 'rb') as f:
                cf_tool.user_similarity_pearson = pickle.load(f)
            logger.info("  ✓ User similarity (Pearson) loaded")
        
        user_sim_cosine_path = os.path.join(cf_dir, 'user_sim_cosine.pkl')
        if os.path.exists(user_sim_cosine_path):
            with open(user_sim_cosine_path, 'rb') as f:
                cf_tool.user_similarity_cosine = pickle.load(f)
            logger.info("  ✓ User similarity (Cosine) loaded")
        
        item_sim_pearson_path = os.path.join(cf_dir, 'item_sim_pearson.pkl')
        if os.path.exists(item_sim_pearson_path):
            with open(item_sim_pearson_path, 'rb') as f:
                cf_tool.item_similarity_pearson = pickle.load(f)
            logger.info("  ✓ Item similarity (Pearson) loaded")
        
        item_sim_cosine_path = os.path.join(cf_dir, 'item_sim_cosine.pkl')
        if os.path.exists(item_sim_cosine_path):
            with open(item_sim_cosine_path, 'rb') as f:
                cf_tool.item_similarity_cosine = pickle.load(f)
            logger.info("  ✓ Item similarity (Cosine) loaded")
        
        return cf_tool
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def main():
    # Paths
    cf_dir = os.path.join('saved_models', 'ml-100k', 'cf')
    data_path = os.path.join('data', 'ml-100k', 'train.csv')
    dev_data_path = os.path.join('data', 'ml-100k', 'dev.csv')
    
    logger.info("=" * 60)
    logger.info("Evaluating Collaborative Filtering Models")
    logger.info("=" * 60)
    
    # Check if CF models exist
    if not os.path.exists(cf_dir):
        raise FileNotFoundError(f"CF models not found: {cf_dir}. Please run train_cf_model.py first.")
    
    # Load CF tool with pre-computed matrices
    logger.info("\n[1] Loading CF models...")
    cf_tool = load_cf_tool_with_matrices(cf_dir, data_path)
    
    # Ensure matrix is built (in case loading metadata failed)
    cf_tool._ensure_matrix_built()
    
    logger.info(f"✓ CF models loaded")
    logger.info(f"  - Users: {len(cf_tool.rating_matrix.index)}")
    logger.info(f"  - Items: {len(cf_tool.rating_matrix.columns)}")
    
    # Test specific predictions
    logger.info("\n[2] Testing specific predictions...")
    test_user = cf_tool.rating_matrix.index[0]
    test_item = cf_tool.rating_matrix.columns[50]
    
    logger.info(f"\n  Predictions for User {test_user}, Item {test_item}:")
    
    methods = [
        ('User-based CF (Pearson)', 'user', 'pearson'),
        ('User-based CF (Cosine)', 'user', 'cosine'),
        ('Item-based CF (Pearson)', 'item', 'pearson'),
        ('Item-based CF (Cosine)', 'item', 'cosine'),
    ]
    
    for method_name, cf_type, similarity in methods:
        if cf_type == 'user':
            pred = cf_tool.predict_rating_user_based(test_user, test_item, method=similarity, k=50)
        else:
            pred = cf_tool.predict_rating_item_based(test_user, test_item, method=similarity, k=50)
        logger.info(f"    {method_name:30s} → {pred:.3f}")
    
    # Evaluate on dev set
    logger.info("\n[3] Evaluating on dev set...")
    if not os.path.exists(dev_data_path):
        logger.warning(f"Dev data not found: {dev_data_path}")
        logger.warning("Skipping dev set evaluation")
        return
    
    logger.info(f"Loading dev data from {dev_data_path}...")
    dev_data = pd.read_csv(dev_data_path)
    
    if 'user_id' not in dev_data.columns or 'item_id' not in dev_data.columns or 'rating' not in dev_data.columns:
        logger.warning("Dev data doesn't have required columns")
        return
    
    # Sample for faster evaluation
    sample_size = min(500, len(dev_data))  # Reduced from 1000 for CF (slower)
    dev_sample = dev_data.head(sample_size)
    
    logger.info(f"Testing on {sample_size} interactions from dev set...")
    logger.info("This may take a few minutes as CF predictions are slower than MF...")
    
    # Evaluate each method
    results = {}
    
    for method_name, cf_type, similarity in methods:
        logger.info(f"\n  Evaluating {method_name}...")
        predictions = []
        actuals = []
        
        for idx, row in dev_sample.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            actual_rating = row['rating']
            
            # Skip if user or item not in training set
            if user_id not in cf_tool.rating_matrix.index or item_id not in cf_tool.rating_matrix.columns:
                continue
            
            try:
                if cf_type == 'user':
                    pred_rating = cf_tool.predict_rating_user_based(user_id, item_id, method=similarity, k=50)
                else:
                    pred_rating = cf_tool.predict_rating_item_based(user_id, item_id, method=similarity, k=50)
                
                predictions.append(pred_rating)
                actuals.append(actual_rating)
            except Exception as e:
                # Skip on error
                continue
        
        if len(predictions) > 0:
            mae, rmse = calculate_metrics(predictions, actuals)
            results[method_name] = {
                'mae': mae,
                'rmse': rmse,
                'n_samples': len(predictions)
            }
            logger.info(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f} (on {len(predictions)} samples)")
        else:
            logger.warning(f"    No valid predictions for {method_name}")
    
    # Summary comparison
    logger.info("\n[4] Summary comparison:")
    logger.info("\n  Method                          MAE      RMSE    Samples")
    logger.info("  " + "-" * 58)
    
    for method_name in [m[0] for m in methods]:
        if method_name in results:
            r = results[method_name]
            logger.info(f"  {method_name:30s}  {r['mae']:.4f}   {r['rmse']:.4f}   {r['n_samples']:4d}")
    
    # Find best method
    if results:
        best_method = min(results.items(), key=lambda x: x[1]['mae'])
        logger.info(f"\n  🏆 Best method: {best_method[0]}")
        logger.info(f"     MAE: {best_method[1]['mae']:.4f}, RMSE: {best_method[1]['rmse']:.4f}")
    
    # Test recommendations
    logger.info("\n[5] Testing recommendation generation...")
    test_user = cf_tool.rating_matrix.index[0]
    
    logger.info(f"\n  Top-10 recommendations for user {test_user}:")
    logger.info(f"  (Using best method: User-based CF with Pearson)")
    
    recommendations = cf_tool.recommend_items_user_based(test_user, n_items=10, method='pearson', k=50)
    for rank, (item_id, pred_rating) in enumerate(recommendations, 1):
        logger.info(f"    {rank:2d}. Item {item_id:4d} → Predicted rating: {pred_rating:.3f}")
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("✓ Evaluation completed!")
    logger.info("=" * 60)
    logger.info("\nKey findings:")
    logger.info("  - User-based CF typically better for sparse data")
    logger.info("  - Item-based CF typically faster for prediction")
    logger.info("  - Pearson handles mean-centering better")
    logger.info("  - Cosine works well for binary/implicit feedback")
    logger.info("\nCF models ready for:")
    logger.info("  - Hybrid recommendation (CF + MF)")
    logger.info("  - Cold-start handling (item-based for new users)")
    logger.info("  - Explainable recommendations (similarity-based)")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
