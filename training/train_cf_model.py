#!/usr/bin/env python3
"""train_cf_model.py

Script to train Collaborative Filtering models for ml-100k dataset.

This script:
1. Loads the ml-100k training data (data/ml-100k/train.csv)
2. Builds the rating matrix
3. Computes user-user similarity matrices (Pearson & Cosine)
4. Computes item-item similarity matrices (Pearson & Cosine)
5. Saves all matrices to saved_models/ml-100k/cf_*.pkl
6. Tests predictions and recommendations

Usage:
    python train_cf_model.py
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from loguru import logger

# Add macrec to path
sys.path.insert(0, 'macrec')

from macrec.tools.collaborative_filtering import CollaborativeFiltering

# Removed duplicate function - using CollaborativeFiltering class instead

def save_similarity_matrices(cf_tool, output_dir):
    """Save pre-computed similarity matrices for faster loading"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save user similarity matrices
    logger.info("Saving user-user similarity matrices...")
    if cf_tool.user_similarity_pearson is not None:
        with open(os.path.join(output_dir, 'user_sim_pearson.pkl'), 'wb') as f:
            pickle.dump(cf_tool.user_similarity_pearson, f)
        logger.info("  ✓ User similarity (Pearson) saved")
    
    if cf_tool.user_similarity_cosine is not None:
        with open(os.path.join(output_dir, 'user_sim_cosine.pkl'), 'wb') as f:
            pickle.dump(cf_tool.user_similarity_cosine, f)
        logger.info("  ✓ User similarity (Cosine) saved")
    
    # Save item similarity matrices
    logger.info("Saving item-item similarity matrices...")
    if cf_tool.item_similarity_pearson is not None:
        with open(os.path.join(output_dir, 'item_sim_pearson.pkl'), 'wb') as f:
            pickle.dump(cf_tool.item_similarity_pearson, f)
        logger.info("  ✓ Item similarity (Pearson) saved")
    
    if cf_tool.item_similarity_cosine is not None:
        with open(os.path.join(output_dir, 'item_sim_cosine.pkl'), 'wb') as f:
            pickle.dump(cf_tool.item_similarity_cosine, f)
        logger.info("  ✓ Item similarity (Cosine) saved")
    
    # Save rating matrix and metadata
    logger.info("Saving rating matrix and metadata...")
    metadata = {
        'rating_matrix': cf_tool.rating_matrix,
        'user_means': cf_tool.user_means,
        'item_means': cf_tool.item_means,
        'data': cf_tool.data
    }
    with open(os.path.join(output_dir, 'cf_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    logger.info("  ✓ Metadata saved")

def main():
    # Configuration
    data_path = os.path.join('data', 'ml-100k', 'train.csv')
    output_dir = os.path.join('saved_models', 'ml-100k', 'cf')

    logger.info("=" * 60)
    logger.info("Collaborative Filtering Model Validation")
    logger.info("=" * 60)

    # Check if models already exist
    model_files = [
        os.path.join(output_dir, 'user_sim_pearson.pkl'),
        os.path.join(output_dir, 'user_sim_cosine.pkl'),
        os.path.join(output_dir, 'item_sim_pearson.pkl'),
        os.path.join(output_dir, 'item_sim_cosine.pkl'),
        os.path.join(output_dir, 'cf_metadata.pkl')
    ]

    all_models_exist = all(os.path.exists(f) for f in model_files)

    if all_models_exist:
        logger.info("✓ All CF models already exist!")
        logger.info("Running validation tests...")
        # Skip training, go directly to testing
        test_existing_models(data_path, output_dir)
        return

    logger.info("Some models missing. Training new models...")

    # Check if data exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    # Create config for CF tool
    cf_config = {
        'data_path': data_path,
        'min_common_items': 2,  # Minimum common items between users
        'min_common_users': 2,  # Minimum common users between items
        'k_neighbors': 50       # Number of neighbors for prediction
    }
    
    # Initialize CF tool
    logger.info("\n[1] Initializing Collaborative Filtering tool...")
    logger.info(f"Loading data from {data_path}...")
    
    # Create temporary config file
    import json
    temp_config_path = 'temp_cf_config.json'
    with open(temp_config_path, 'w') as f:
        json.dump(cf_config, f)
    
    try:
        cf_tool = CollaborativeFiltering(config_path=temp_config_path)
        logger.info(f"✓ Data loaded: {len(cf_tool.data)} interactions")
        logger.info(f"  - Users: {len(cf_tool.rating_matrix.index)}")
        logger.info(f"  - Items: {len(cf_tool.rating_matrix.columns)}")
        logger.info(f"  - Sparsity: {(1 - len(cf_tool.data) / (len(cf_tool.rating_matrix.index) * len(cf_tool.rating_matrix.columns))) * 100:.2f}%")
    finally:
        # Clean up temp config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    
    # Compute similarity matrices
    logger.info("\n[2] Computing similarity matrices...")
    logger.info("This may take several minutes depending on dataset size...")
    
    # User-based similarities
    logger.info("\n  Computing user-user similarities...")
    logger.info("    - Pearson correlation...")
    cf_tool.compute_user_similarity_pearson()
    logger.info("    - Cosine similarity...")
    cf_tool.compute_user_similarity_cosine()
    
    # Item-based similarities
    logger.info("\n  Computing item-item similarities...")
    logger.info("    - Pearson correlation...")
    cf_tool.compute_item_similarity_pearson()
    logger.info("    - Cosine similarity...")
    cf_tool.compute_item_similarity_cosine()
    
    logger.info("\n✓ All similarity matrices computed!")
    
    # Save matrices
    logger.info("\n[3] Saving similarity matrices...")
    save_similarity_matrices(cf_tool, output_dir)
    logger.info(f"✓ All matrices saved to {output_dir}")
    
    # Test predictions
    logger.info("\n[4] Testing CF predictions...")
    
    # Test user-based CF
    test_user = cf_tool.rating_matrix.index[0]
    test_item = cf_tool.rating_matrix.columns[0]
    
    logger.info(f"\n  User-based CF (Pearson):")
    pred_user_pearson = cf_tool.predict_rating_user_based(test_user, test_item, method='pearson', k=50)
    logger.info(f"    User {test_user}, Item {test_item} → Predicted rating: {pred_user_pearson:.3f}")
    
    logger.info(f"\n  User-based CF (Cosine):")
    pred_user_cosine = cf_tool.predict_rating_user_based(test_user, test_item, method='cosine', k=50)
    logger.info(f"    User {test_user}, Item {test_item} → Predicted rating: {pred_user_cosine:.3f}")
    
    # Test item-based CF
    logger.info(f"\n  Item-based CF (Pearson):")
    pred_item_pearson = cf_tool.predict_rating_item_based(test_user, test_item, method='pearson', k=50)
    logger.info(f"    User {test_user}, Item {test_item} → Predicted rating: {pred_item_pearson:.3f}")
    
    logger.info(f"\n  Item-based CF (Cosine):")
    pred_item_cosine = cf_tool.predict_rating_item_based(test_user, test_item, method='cosine', k=50)
    logger.info(f"    User {test_user}, Item {test_item} → Predicted rating: {pred_item_cosine:.3f}")
    
    # Test recommendations
    logger.info("\n[5] Testing recommendations...")
    logger.info(f"\n  Top-10 recommendations for user {test_user} (User-based CF, Pearson):")
    recommendations = cf_tool.recommend_items_user_based(test_user, n_items=10, method='pearson', k=50)
    for rank, (item_id, pred_rating) in enumerate(recommendations, 1):
        logger.info(f"    {rank:2d}. Item {item_id:4d} → Predicted rating: {pred_rating:.3f}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✓ Collaborative Filtering training completed!")
    logger.info("=" * 60)
    logger.info("\nSaved files:")
    logger.info(f"  - {os.path.join(output_dir, 'user_sim_pearson.pkl')}")
    logger.info(f"  - {os.path.join(output_dir, 'user_sim_cosine.pkl')}")
    logger.info(f"  - {os.path.join(output_dir, 'item_sim_pearson.pkl')}")
    logger.info(f"  - {os.path.join(output_dir, 'item_sim_cosine.pkl')}")
    logger.info(f"  - {os.path.join(output_dir, 'cf_metadata.pkl')}")
    logger.info("\nCF models ready for use in:")
    logger.info("  - Searcher agent (CF recommendations)")
    logger.info("  - Hybrid recommendation systems")
    logger.info("  - Similarity-based filtering")
    logger.info("=" * 60)

def test_existing_models(data_path, output_dir):
    """Test existing CF models without retraining"""

    # Create config for CF tool
    cf_config = {
        'data_path': data_path,
        'model_path': os.path.join('saved_models', 'ml-100k'),
        'min_common_items': 2,
        'min_common_users': 2,
        'k_neighbors': 50
    }

    # Create temporary config file
    temp_config_path = 'temp_cf_config.json'
    import json
    with open(temp_config_path, 'w') as f:
        json.dump(cf_config, f)

    try:
        cf_tool = CollaborativeFiltering(config_path=temp_config_path)
        logger.info(f"✓ CF tool initialized with pre-computed models")
        logger.info(f"  - Data loaded: {len(cf_tool.data)} interactions")
        logger.info(f"  - Users: {len(cf_tool.rating_matrix.index)}")
        logger.info(f"  - Items: {len(cf_tool.rating_matrix.columns)}")
        logger.info(f"  - User similarity (Pearson): {'✓ Loaded' if cf_tool.user_similarity_pearson is not None else '✗ Not loaded'}")
        logger.info(f"  - User similarity (Cosine): {'✓ Loaded' if cf_tool.user_similarity_cosine is not None else '✗ Not loaded'}")
        logger.info(f"  - Item similarity (Pearson): {'✓ Loaded' if cf_tool.item_similarity_pearson is not None else '✗ Not loaded'}")
        logger.info(f"  - Item similarity (Cosine): {'✓ Loaded' if cf_tool.item_similarity_cosine is not None else '✗ Not loaded'}")
    finally:
        # Clean up temp config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    # Test predictions
    logger.info("\nTesting CF predictions...")

    # Test user-based CF
    test_user = cf_tool.rating_matrix.index[0]
    test_item = cf_tool.rating_matrix.columns[0]

    logger.info(f"\n  User-based CF (Pearson):")
    pred_user_pearson = cf_tool.predict_rating_user_based(test_user, test_item, method='pearson', k=50)
    logger.info(f"    User {test_user}, Item {test_item} → Predicted rating: {pred_user_pearson:.3f}")

    logger.info(f"\n  User-based CF (Cosine):")
    pred_user_cosine = cf_tool.predict_rating_user_based(test_user, test_item, method='cosine', k=50)
    logger.info(f"    User {test_user}, Item {test_item} → Predicted rating: {pred_user_cosine:.3f}")

    # Test recommendations
    logger.info("\nTesting recommendations...")
    logger.info(f"\n  Top-5 recommendations for user {test_user} (User-based CF, Pearson):")
    recommendations = cf_tool.recommend_items_user_based(test_user, n_items=5, method='pearson', k=50)
    for rank, (item_id, pred_rating) in enumerate(recommendations, 1):
        logger.info(f"    {rank}. Item {item_id:4d} → Predicted rating: {pred_rating:.3f}")

    logger.info("\n" + "=" * 60)
    logger.info("✓ CF model validation completed!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
