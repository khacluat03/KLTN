#!/usr/bin/env python3
"""train_rating_model.py

Script to train the Matrix Factorization rating model for ml-100k dataset.

This script:
1. Loads the ml-100k training data (data/ml-100k/train.csv)
2. Trains a Matrix Factorization model using SVD
3. Saves the trained model to saved_models/ml-100k/mf_model.pkl
4. Performs a sample prediction to verify the model works

Usage:
    python train_rating_model.py
"""

import os
import pandas as pd
from pathlib import Path
from loguru import logger

from macrec.systems.methods.matrix_factorization import MatrixFactorization

def main():
    # Configuration for ml-100k dataset
    data_path = os.path.join('data', 'ml-100k', 'train.csv')
    
    # Ensure saved_models/ml-100k directory exists
    model_dir = Path('saved_models') / 'ml-100k'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Model path - always saved to saved_models/ml-100k/mf_model.pkl
    model_path = str(model_dir / 'mf_model.pkl')
    n_factors = 50  # Number of latent factors
    
    # Check if data file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    # Check if model already exists
    if os.path.exists(model_path):
        logger.info(f"Model already exists at {model_path}")
        logger.info("Loading existing model...")
        model = MatrixFactorization(n_factors=n_factors)
        model.load(model_path)
        logger.info("✓ Model loaded successfully")
    else:
        logger.info(f"Training new Matrix Factorization model on {data_path}...")
        
        # Load training data
        logger.info("Loading training data...")
        data = pd.read_csv(data_path)
        
        # Check required columns
        required_cols = ['user_id', 'item_id', 'rating']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must have columns: {required_cols}")
        
        logger.info(f"Data loaded: {len(data)} interactions, {data['user_id'].nunique()} users, {data['item_id'].nunique()} items")
        
        # Train model
        model = MatrixFactorization(n_factors=n_factors)
        model.fit(data)
        
        # Save model
        logger.info(f"Saving model to {model_path}...")
        model.save(model_path)
        logger.info("✓ Model trained and saved successfully")
        logger.info(f"✓ Model directory verified: {model_dir.absolute()}")
    
    # Test prediction with sample user/item
    # Use first user and first item from the model's known IDs
    if model.user_ids and model.item_ids:
        sample_user = model.user_ids[0]
        sample_item = model.item_ids[0]
        
        try:
            rating = model.predict(sample_user, sample_item)
            logger.info(f"✓ Sample prediction - user {sample_user}, item {sample_item}: {rating:.3f}")
        except Exception as e:
            logger.error(f"✗ Prediction test failed: {e}")
    else:
        logger.warning("No user/item IDs available for testing")
    
    logger.info("=" * 60)
    logger.info("Training complete! Model ready for use.")
    logger.info(f"Model location: {model_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
