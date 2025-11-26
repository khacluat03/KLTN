#!/usr/bin/env python3
"""evaluate_cf_standalone.py

Standalone script to evaluate CF models without importing macrec modules.

Usage:
    python evaluate_cf_standalone.py
"""

import os
import pandas as pd
import numpy as np
import pickle
from loguru import logger

def calculate_metrics(predictions, actuals):
    """Calculate MAE and RMSE"""
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    return mae, rmse

def predict_rating_user_based(user_id, item_id, rating_matrix, user_sim_matrix, user_means, k=50):
    """Predict rating using user-based CF"""
    if user_id not in rating_matrix.index or item_id not in rating_matrix.columns:
        return rating_matrix.values[rating_matrix.values > 0].mean()
    
    # Check if already rated
    if rating_matrix.loc[user_id, item_id] != 0:
        return rating_matrix.loc[user_id, item_id]
    
    # Get users who rated this item
    item_ratings = rating_matrix[item_id]
    rated_users = item_ratings[item_ratings != 0].index
    
    if len(rated_users) == 0:
        return user_means.loc[user_id]
    
    # Get similarities
    user_similarities = user_sim_matrix.loc[user_id, rated_users]
    
    # Filter positive similarities
    positive_mask = user_similarities > 0
    if positive_mask.sum() == 0:
        return user_means.loc[user_id]
    
    user_similarities = user_similarities[positive_mask]
    rated_users = rated_users[positive_mask]
    
    # Select top k neighbors
    if k is not None and k < len(rated_users):
        top_k_indices = user_similarities.nlargest(k).index
        user_similarities = user_similarities[top_k_indices]
        rated_users = top_k_indices
    
    # Predict
    user_mean = user_means.loc[user_id]
    numerator = 0.0
    denominator = 0.0
    
    for other_user in rated_users:
        sim = user_similarities.loc[other_user]
        other_user_mean = user_means.loc[other_user]
        rating = rating_matrix.loc[other_user, item_id]
        
        numerator += sim * (rating - other_user_mean)
        denominator += abs(sim)
    
    if denominator == 0:
        return user_mean
    
    predicted_rating = user_mean + (numerator / denominator)
    return np.clip(predicted_rating, 1.0, 5.0)

def predict_rating_item_based(user_id, item_id, rating_matrix, item_sim_matrix, item_means, k=50):
    """Predict rating using item-based CF"""
    if user_id not in rating_matrix.index or item_id not in rating_matrix.columns:
        return rating_matrix.values[rating_matrix.values > 0].mean()
    
    # Check if already rated
    if rating_matrix.loc[user_id, item_id] != 0:
        return rating_matrix.loc[user_id, item_id]
    
    # Get items rated by this user
    user_ratings = rating_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings != 0].index
    
    if len(rated_items) == 0:
        return item_means.loc[item_id]
    
    # Get similarities
    item_similarities = item_sim_matrix.loc[item_id, rated_items]
    
    # Filter positive similarities
    positive_mask = item_similarities > 0
    if positive_mask.sum() == 0:
        return item_means.loc[item_id]
    
    item_similarities = item_similarities[positive_mask]
    rated_items = rated_items[positive_mask]
    
    # Select top k neighbors
    if k is not None and k < len(rated_items):
        top_k_indices = item_similarities.nlargest(k).index
        item_similarities = item_similarities[top_k_indices]
        rated_items = top_k_indices
    
    # Predict
    item_mean = item_means.loc[item_id]
    numerator = 0.0
    denominator = 0.0
    
    for other_item in rated_items:
        sim = item_similarities.loc[other_item]
        other_item_mean = item_means.loc[other_item]
        rating = rating_matrix.loc[user_id, other_item]
        
        numerator += sim * (rating - other_item_mean)
        denominator += abs(sim)
    
    if denominator == 0:
        return item_mean
    
    predicted_rating = item_mean + (numerator / denominator)
    return np.clip(predicted_rating, 1.0, 5.0)

def main():
    # Paths
    cf_dir = os.path.join('saved_models', 'ml-100k', 'cf')
    dev_data_path = os.path.join('data', 'ml-100k', 'dev.csv')
    
    logger.info("=" * 60)
    logger.info("Evaluating Collaborative Filtering Models")
    logger.info("=" * 60)
    
    # Load CF models
    logger.info("\n[1] Loading CF models...")
    
    with open(os.path.join(cf_dir, 'cf_metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    rating_matrix = metadata['rating_matrix']
    user_means = metadata['user_means']
    item_means = metadata['item_means']
    
    with open(os.path.join(cf_dir, 'user_sim_pearson.pkl'), 'rb') as f:
        user_sim_pearson = pickle.load(f)
    
    with open(os.path.join(cf_dir, 'user_sim_cosine.pkl'), 'rb') as f:
        user_sim_cosine = pickle.load(f)
    
    with open(os.path.join(cf_dir, 'item_sim_pearson.pkl'), 'rb') as f:
        item_sim_pearson = pickle.load(f)
    
    with open(os.path.join(cf_dir, 'item_sim_cosine.pkl'), 'rb') as f:
        item_sim_cosine = pickle.load(f)
    
    logger.info("✓ CF models loaded")
    logger.info(f"  - Users: {len(rating_matrix.index)}")
    logger.info(f"  - Items: {len(rating_matrix.columns)}")
    
    # Load dev data
    logger.info(f"\n[2] Loading dev data from {dev_data_path}...")
    dev_data = pd.read_csv(dev_data_path)
    
    # Sample for evaluation
    sample_size = min(500, len(dev_data))
    dev_sample = dev_data.head(sample_size)
    logger.info(f"Testing on {sample_size} interactions from dev set...")
    
    # Evaluate each method
    methods = [
        ('User-based CF (Pearson)', 'user', user_sim_pearson),
        ('User-based CF (Cosine)', 'user', user_sim_cosine),
        ('Item-based CF (Pearson)', 'item', item_sim_pearson),
        ('Item-based CF (Cosine)', 'item', item_sim_cosine),
    ]
    
    results = {}
    
    for method_name, cf_type, sim_matrix in methods:
        logger.info(f"\n[3] Evaluating {method_name}...")
        predictions = []
        actuals = []
        
        for idx, row in dev_sample.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            actual_rating = row['rating']
            
            if user_id not in rating_matrix.index or item_id not in rating_matrix.columns:
                continue
            
            try:
                if cf_type == 'user':
                    pred_rating = predict_rating_user_based(
                        user_id, item_id, rating_matrix, sim_matrix, user_means, k=50
                    )
                else:
                    pred_rating = predict_rating_item_based(
                        user_id, item_id, rating_matrix, sim_matrix, item_means, k=50
                    )
                
                predictions.append(pred_rating)
                actuals.append(actual_rating)
            except Exception as e:
                continue
        
        if len(predictions) > 0:
            mae, rmse = calculate_metrics(predictions, actuals)
            results[method_name] = {
                'mae': mae,
                'rmse': rmse,
                'n_samples': len(predictions)
            }
            logger.info(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f} (on {len(predictions)} samples)")
    
    # Summary
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
    
    # Compare with MF baseline
    logger.info("\n[5] Comparison with Matrix Factorization:")
    logger.info("  MF model:                      MAE: 0.8790, RMSE: 1.1141")
    if results:
        best_cf_mae = best_method[1]['mae']
        best_cf_rmse = best_method[1]['rmse']
        logger.info(f"  Best CF:                       MAE: {best_cf_mae:.4f}, RMSE: {best_cf_rmse:.4f}")
        
        if best_cf_mae < 0.88:
            logger.info("  → CF performs BETTER than MF! ✅")
        elif best_cf_mae < 1.0:
            logger.info("  → CF performs comparably to MF ⭐")
        else:
            logger.info("  → MF performs better than CF")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Evaluation completed!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
