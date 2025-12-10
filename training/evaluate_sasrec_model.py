#!/usr/bin/env python3
"""evaluate_sasrec_model.py

Script to evaluate the trained SASRec model.

This script:
1. Loads the trained SASRec model
2. Tests specific predictions
3. Evaluates Hit@10 and NDCG@10 on test users
4. Generates recommendations

Usage:
    python training/evaluate_sasrec_model.py
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from macrec.systems.methods.sasrec import load_sasrec

def calculate_metrics(model, user_sequences, test_users, item_id_map, k=10):
    """Calculate Hit@K and NDCG@K"""
    hits = 0
    ndcgs = 0
    n_test = 0
    
    all_items = list(item_id_map.values())
    
    for user_id in test_users:
        if user_id not in user_sequences:
            continue
            
        seq = user_sequences[user_id]
        if len(seq) < 2:
            continue
            
        # Use all but last item as input, try to predict last item
        input_seq = seq[:-1]
        target_item = seq[-1]
        
        # Truncate if needed
        if len(input_seq) > model.max_len:
            input_seq = input_seq[-model.max_len:]
            
        # Prepare input
        seq_tensor = torch.tensor([input_seq], dtype=torch.long, device=model.device)
        
        # Predict
        with torch.no_grad():
            # Score all items
            scores = model.predict(seq_tensor)  # (1, n_items)
            scores = scores[0].cpu().numpy()
            
        # Rank items
        # Exclude items already in sequence (except target) to avoid recommending them
        # This is more realistic for recommendation scenarios
        
        # Get items already in sequence (excluding target)
        seen_items = set(input_seq)
        
        # Mask out seen items by setting their scores to -inf
        # scores indices: 0 -> item ID 1, 1 -> item ID 2, etc.
        for idx in range(len(scores)):
            item_id = idx + 1  # Convert index to item ID
            if item_id in seen_items:
                scores[idx] = -np.inf
        
        # Get top K indices
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        # Adjust indices to match item IDs
        top_k_items = [idx + 1 for idx in top_k_indices]
        
        if target_item in top_k_items:
            hits += 1
            
            # Calculate NDCG
            rank = top_k_items.index(target_item)
            ndcgs += 1.0 / np.log2(rank + 2)
            
        n_test += 1
        
        if n_test % 100 == 0:
            logger.info(f"Evaluated {n_test} users...")
            
    return hits / n_test if n_test > 0 else 0, ndcgs / n_test if n_test > 0 else 0

def main():
    # Paths
    model_path = os.path.join('saved_models', 'sasrec_ml-100k.pkl')
    
    logger.info("=" * 60)
    logger.info("Evaluating SASRec Model")
    logger.info("=" * 60)
    
    # 1. Load model
    logger.info("\n[1] Loading trained model...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Please run train_sasrec.py first.")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, item_id_map, user_sequences = load_sasrec(model_path, device=device)
    
    logger.info(f"✓ Model loaded from {model_path}")
    logger.info(f"  - Items: {model.n_items}")
    logger.info(f"  - Max sequence length: {model.max_len}")
    logger.info(f"  - Device: {device}")
    
    # 2. Test specific prediction
    logger.info("\n[2] Testing specific prediction...")
    test_user = list(user_sequences.keys())[0]
    seq = user_sequences[test_user]
    
    logger.info(f"  User {test_user}")
    logger.info(f"  Sequence: {seq}")
    
    input_seq = seq[:-1]
    if len(input_seq) > model.max_len:
        input_seq = input_seq[-model.max_len:]
        
    seq_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)
    
    with torch.no_grad():
        scores = model.predict(seq_tensor)
        scores = scores[0].cpu().numpy()
        
    # Get top 5
    top_5_indices = np.argsort(scores)[-5:][::-1]
    top_5_items = [idx + 1 for idx in top_5_indices]
    
    logger.info(f"  Top 5 predictions: {top_5_items}")
    logger.info(f"  Actual next item: {seq[-1]}")
    
    # 3. Evaluate on test users (last item split)
    logger.info("\n[3] Evaluating Hit@10 and NDCG@10...")
    
    # Use all users for evaluation
    test_users = list(user_sequences.keys())
    # Sample if too many
    if len(test_users) > 1000:
        import random
        random.seed(42)
        test_users = random.sample(test_users, 1000)
        logger.info(f"  Sampling {1000} users for evaluation...")
    
    hit_10, ndcg_10 = calculate_metrics(model, user_sequences, test_users, item_id_map, k=10)
    
    logger.info(f"\n  Evaluation Results (on {len(test_users)} users):")
    logger.info(f"  - Hit@10:  {hit_10:.4f}")
    logger.info(f"  - NDCG@10: {ndcg_10:.4f}")
    
    # 4. Generate recommendations for a user
    logger.info("\n[4] Generating recommendations...")
    
    # Predict for the full sequence (to recommend NEXT item after the known sequence)
    full_seq = user_sequences[test_user]
    if len(full_seq) > model.max_len:
        full_seq = full_seq[-model.max_len:]
        
    seq_tensor = torch.tensor([full_seq], dtype=torch.long, device=device)
    
    with torch.no_grad():
        scores = model.predict(seq_tensor)
        scores = scores[0].cpu().numpy()
        
    top_10_indices = np.argsort(scores)[-10:][::-1]
    top_10_items = [idx + 1 for idx in top_10_indices]
    
    logger.info(f"  Top 10 recommendations for User {test_user} (future):")
    for i, item in enumerate(top_10_items, 1):
        logger.info(f"    {i}. Item {item}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✓ Evaluation completed!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
