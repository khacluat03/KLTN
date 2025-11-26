"""
Demo script cho Collaborative Filtering
Min họa cách sử dụng các phương pháp:
- User-based CF với Pearson Correlation
- User-based CF với Cosine Similarity
- Item-based CF với Pearson Correlation
- Item-based CF với Cosine Similarity
"""

import sys
import os

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macrec.tools.collaborative_filtering import CollaborativeFiltering
from loguru import logger

def demo_collaborative_filtering():
    """Demo các phương pháp Collaborative Filtering"""
    
    # Khởi tạo CF tool
    config_path = 'config/tools/collaborative_filtering.json'
    logger.info(f"Loading Collaborative Filtering from {config_path}")
    
    try:
        cf = CollaborativeFiltering(config_path=config_path)
    except Exception as e:
        logger.error(f"Error loading CF tool: {e}")
        logger.info("Make sure data file exists at the path specified in config")
        return
    
    # Demo 1: Dự đoán rating với User-based CF
    logger.info("\n" + "="*60)
    logger.info("DEMO 1: Dự đoán rating với User-based CF")
    logger.info("="*60)
    
    user_id = 1
    item_id = 100
    
    # Pearson Correlation
    rating_user_pearson = cf.predict_rating_user_based(
        user_id=user_id,
        item_id=item_id,
        method='pearson',
        k=50
    )
    logger.info(f"User {user_id} - Item {item_id} (User-based, Pearson): {rating_user_pearson:.4f}")
    
    # Cosine Similarity
    rating_user_cosine = cf.predict_rating_user_based(
        user_id=user_id,
        item_id=item_id,
        method='cosine',
        k=50
    )
    logger.info(f"User {user_id} - Item {item_id} (User-based, Cosine): {rating_user_cosine:.4f}")
    
    # Demo 2: Dự đoán rating với Item-based CF
    logger.info("\n" + "="*60)
    logger.info("DEMO 2: Dự đoán rating với Item-based CF")
    logger.info("="*60)
    
    # Pearson Correlation
    rating_item_pearson = cf.predict_rating_item_based(
        user_id=user_id,
        item_id=item_id,
        method='pearson',
        k=50
    )
    logger.info(f"User {user_id} - Item {item_id} (Item-based, Pearson): {rating_item_pearson:.4f}")
    
    # Cosine Similarity
    rating_item_cosine = cf.predict_rating_item_based(
        user_id=user_id,
        item_id=item_id,
        method='cosine',
        k=50
    )
    logger.info(f"User {user_id} - Item {item_id} (Item-based, Cosine): {rating_item_cosine:.4f}")
    
    # Demo 3: Recommend items với User-based CF
    logger.info("\n" + "="*60)
    logger.info("DEMO 3: Recommend items với User-based CF")
    logger.info("="*60)
    
    recommendations_user = cf.recommend_items_user_based(
        user_id=user_id,
        n_items=10,
        method='pearson',
        k=50
    )
    
    logger.info(f"Top 10 recommendations cho User {user_id} (User-based, Pearson):")
    for i, (item_id, rating) in enumerate(recommendations_user, 1):
        logger.info(f"  {i}. Item {item_id}: predicted rating = {rating:.4f}")
    
    # Demo 4: Recommend items với Item-based CF
    logger.info("\n" + "="*60)
    logger.info("DEMO 4: Recommend items với Item-based CF")
    logger.info("="*60)
    
    recommendations_item = cf.recommend_items_item_based(
        user_id=user_id,
        n_items=10,
        method='cosine',
        k=50
    )
    
    logger.info(f"Top 10 recommendations cho User {user_id} (Item-based, Cosine):")
    for i, (item_id, rating) in enumerate(recommendations_item, 1):
        logger.info(f"  {i}. Item {item_id}: predicted rating = {rating:.4f}")
    
    # Demo 5: Tìm users tương tự
    logger.info("\n" + "="*60)
    logger.info("DEMO 5: Tìm users tương tự")
    logger.info("="*60)
    
    similar_users_info = cf.get_similarity_info(
        entity_id=user_id,
        entity_type='user',
        method='pearson',
        top_k=5
    )
    logger.info(similar_users_info)
    
    # Demo 6: Tìm items tương tự
    logger.info("\n" + "="*60)
    logger.info("DEMO 6: Tìm items tương tự")
    logger.info("="*60)
    
    similar_items_info = cf.get_similarity_info(
        entity_id=item_id,
        entity_type='item',
        method='cosine',
        top_k=5
    )
    logger.info(similar_items_info)
    
    # Demo 7: So sánh các phương pháp
    logger.info("\n" + "="*60)
    logger.info("DEMO 7: So sánh các phương pháp")
    logger.info("="*60)
    
    test_cases = [
        (1, 50),
        (1, 100),
        (2, 50),
        (2, 100),
    ]
    
    logger.info(f"{'User':<6} {'Item':<6} {'User-Pearson':<15} {'User-Cosine':<15} {'Item-Pearson':<15} {'Item-Cosine':<15}")
    logger.info("-" * 80)
    
    for u_id, i_id in test_cases:
        try:
            up = cf.predict_rating_user_based(u_id, i_id, 'pearson', k=30)
            uc = cf.predict_rating_user_based(u_id, i_id, 'cosine', k=30)
            ip = cf.predict_rating_item_based(u_id, i_id, 'pearson', k=30)
            ic = cf.predict_rating_item_based(u_id, i_id, 'cosine', k=30)
            
            logger.info(f"{u_id:<6} {i_id:<6} {up:<15.4f} {uc:<15.4f} {ip:<15.4f} {ic:<15.4f}")
        except Exception as e:
            logger.warning(f"Error for user {u_id}, item {i_id}: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("Demo hoàn thành!")
    logger.info("="*60)

if __name__ == '__main__':
    demo_collaborative_filtering()



