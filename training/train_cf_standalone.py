#!/usr/bin/env python3
"""train_cf_standalone.py

Standalone script to compute CF similarity matrices without importing macrec modules.
This avoids circular import issues.

Usage:
    python train_cf_standalone.py
"""

import os
import pandas as pd
import numpy as np
import pickle
from loguru import logger

def pearson_correlation(vec1, vec2, min_common=2):
    """Calculate Pearson correlation between two vectors"""
    common_mask = (vec1 != 0) & (vec2 != 0)
    
    if common_mask.sum() < min_common:
        return 0.0
    
    vec1_common = vec1[common_mask]
    vec2_common = vec2[common_mask]
    
    mean1 = vec1_common.mean()
    mean2 = vec2_common.mean()
    
    numerator = ((vec1_common - mean1) * (vec2_common - mean2)).sum()
    denominator1 = ((vec1_common - mean1) ** 2).sum()
    denominator2 = ((vec2_common - mean2) ** 2).sum()
    
    denominator = np.sqrt(denominator1 * denominator2)
    
    if denominator == 0:
        return 0.0
    
    correlation = numerator / denominator
    return np.clip(correlation, -1.0, 1.0)

def cosine_similarity(vec1, vec2, min_common=2):
    """Calculate Cosine similarity between two vectors"""
    common_mask = (vec1 != 0) & (vec2 != 0)
    
    if common_mask.sum() < min_common:
        return 0.0
    
    vec1_common = vec1[common_mask]
    vec2_common = vec2[common_mask]
    
    dot_product = (vec1_common * vec2_common).sum()
    norm1 = np.sqrt((vec1_common ** 2).sum())
    norm2 = np.sqrt((vec2_common ** 2).sum())
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return np.clip(similarity, 0.0, 1.0)

def compute_similarity_matrix(matrix, similarity_func, entity_type='user'):
    """Compute similarity matrix"""
    n = len(matrix)
    similarity_matrix = pd.DataFrame(
        index=matrix.index,
        columns=matrix.index,
        dtype=float
    )
    
    logger.info(f"Computing {entity_type}-{entity_type} similarity matrix...")
    for i, entity1 in enumerate(matrix.index):
        if i % 100 == 0:
            logger.info(f"  Processing {entity_type} {i+1}/{n}")
        for entity2 in matrix.index:
            if entity1 == entity2:
                similarity_matrix.loc[entity1, entity2] = 1.0
            elif pd.isna(similarity_matrix.loc[entity2, entity1]):
                sim = similarity_func(matrix.loc[entity1], matrix.loc[entity2])
                similarity_matrix.loc[entity1, entity2] = sim
                similarity_matrix.loc[entity2, entity1] = sim
            else:
                similarity_matrix.loc[entity1, entity2] = similarity_matrix.loc[entity2, entity1]
    
    return similarity_matrix

def main():
    # Configuration
    data_path = os.path.join('data', 'ml-100k', 'train.csv')
    output_dir = os.path.join('saved_models', 'ml-100k', 'cf')
    
    logger.info("=" * 60)
    logger.info("Training Collaborative Filtering Models")
    logger.info("=" * 60)
    
    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    logger.info(f"\n[1] Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    logger.info(f"✓ Data loaded: {len(data)} interactions")
    
    # Build rating matrix
    logger.info("\n[2] Building rating matrix...")
    rating_matrix = data.pivot_table(
        index='user_id',
        columns='item_id',
        values='rating',
        fill_value=0
    )
    logger.info(f"✓ Rating matrix built: {rating_matrix.shape[0]} users × {rating_matrix.shape[1]} items")
    logger.info(f"  Sparsity: {(1 - len(data) / (rating_matrix.shape[0] * rating_matrix.shape[1])) * 100:.2f}%")
    
    # Compute user similarities
    logger.info("\n[3] Computing user-user similarities...")
    logger.info("  This may take 10-15 minutes...")
    
    logger.info("\n  Computing Pearson correlation...")
    user_sim_pearson = compute_similarity_matrix(rating_matrix, pearson_correlation, 'user')
    
    logger.info("\n  Computing Cosine similarity...")
    user_sim_cosine = compute_similarity_matrix(rating_matrix, cosine_similarity, 'user')
    
    # Compute item similarities
    logger.info("\n[4] Computing item-item similarities...")
    logger.info("  This may take 10-15 minutes...")
    
    item_matrix = rating_matrix.T
    
    logger.info("\n  Computing Pearson correlation...")
    item_sim_pearson = compute_similarity_matrix(item_matrix, pearson_correlation, 'item')
    
    logger.info("\n  Computing Cosine similarity...")
    item_sim_cosine = compute_similarity_matrix(item_matrix, cosine_similarity, 'item')
    
    # Save matrices
    logger.info("\n[5] Saving similarity matrices...")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'user_sim_pearson.pkl'), 'wb') as f:
        pickle.dump(user_sim_pearson, f)
    logger.info("  ✓ User similarity (Pearson) saved")
    
    with open(os.path.join(output_dir, 'user_sim_cosine.pkl'), 'wb') as f:
        pickle.dump(user_sim_cosine, f)
    logger.info("  ✓ User similarity (Cosine) saved")
    
    with open(os.path.join(output_dir, 'item_sim_pearson.pkl'), 'wb') as f:
        pickle.dump(item_sim_pearson, f)
    logger.info("  ✓ Item similarity (Pearson) saved")
    
    with open(os.path.join(output_dir, 'item_sim_cosine.pkl'), 'wb') as f:
        pickle.dump(item_sim_cosine, f)
    logger.info("  ✓ Item similarity (Cosine) saved")
    
    # Save metadata
    metadata = {
        'rating_matrix': rating_matrix,
        'user_means': rating_matrix.mean(axis=1),
        'item_means': rating_matrix.mean(axis=0),
        'data': data
    }
    with open(os.path.join(output_dir, 'cf_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    logger.info("  ✓ Metadata saved")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✓ Collaborative Filtering training completed!")
    logger.info("=" * 60)
    logger.info(f"\nSaved files in {output_dir}:")
    logger.info("  - user_sim_pearson.pkl")
    logger.info("  - user_sim_cosine.pkl")
    logger.info("  - item_sim_pearson.pkl")
    logger.info("  - item_sim_cosine.pkl")
    logger.info("  - cf_metadata.pkl")
    logger.info("\nCF models ready for use!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
