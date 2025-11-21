"""
Script để tạo database SQLite từ dữ liệu ml-100k dataset
Tạo 3 bảng: users, items và interactions
"""
import sqlite3
import pandas as pd
import os
from loguru import logger

DEFAULT_DATASET_CONFIG = {
    "beauty": {
        "db_path": "data/Beauty/database.db",
        "data_dir": "data/Beauty",
    },
    "ml-100k": {
        "db_path": "data/ml-100k/database.db",
        "data_dir": "data/ml-100k",
    },
}


def resolve_paths(db_path: str | None, data_dir: str | None, dataset: str) -> tuple[str, str]:
    dataset_key = dataset.lower()
    if dataset_key not in DEFAULT_DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset '{dataset}'. Supported datasets: {', '.join(DEFAULT_DATASET_CONFIG)}")
    defaults = DEFAULT_DATASET_CONFIG[dataset_key]
    resolved_db_path = db_path or defaults["db_path"]
    resolved_data_dir = data_dir or defaults["data_dir"]
    return resolved_db_path, resolved_data_dir


def create_database(db_path: str | None = None, data_dir: str | None = None, dataset: str = "ml-100k"):
    """
    Tạo database SQLite từ dữ liệu ml-100k
    
    Args:
        db_path: Đường dẫn đến file database SQLite
        data_dir: Thư mục chứa các file CSV
        dataset: Tên dataset (ml-100k hoặc beauty)
    """
    db_path, data_dir = resolve_paths(db_path, data_dir, dataset)
    dataset_key = dataset.lower()

    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Xóa database cũ nếu có
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info(f"Removed existing database: {db_path}")
    
    # Kết nối database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    logger.info("Creating database tables...")
    
    if dataset_key == "ml-100k":
        # Tạo bảng users cho ml-100k
        cursor.execute('''
            CREATE TABLE users (
                user_id INTEGER PRIMARY KEY,
                age INTEGER,
                gender TEXT,
                occupation TEXT,
                user_profile TEXT
            )
        ''')
        logger.info("Created table: users")
        
        # Tạo bảng items cho ml-100k
        cursor.execute('''
            CREATE TABLE items (
                item_id INTEGER PRIMARY KEY,
                title TEXT,
                release_date TEXT,
                video_release_date TEXT,
                genre TEXT,
                item_attributes TEXT
            )
        ''')
        logger.info("Created table: items")
        
        # Tạo bảng interactions cho ml-100k
        cursor.execute('''
            CREATE TABLE interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                item_id INTEGER NOT NULL,
                rating REAL NOT NULL,
                timestamp INTEGER,
                dataset_split TEXT,
                neg_item_id TEXT,
                position INTEGER,
                history_item_id TEXT,
                history_rating TEXT,
                history TEXT,
                user_profile TEXT,
                target_item_attributes TEXT,
                candidate_item_id TEXT,
                candidate_item_attributes TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (item_id) REFERENCES items(item_id)
            )
        ''')
        logger.info("Created table: interactions")
        
        # Tạo index để tăng tốc query
        cursor.execute('CREATE INDEX idx_interactions_user_id ON interactions(user_id)')
        cursor.execute('CREATE INDEX idx_interactions_item_id ON interactions(item_id)')
        cursor.execute('CREATE INDEX idx_interactions_rating ON interactions(rating)')
        cursor.execute('CREATE INDEX idx_interactions_timestamp ON interactions(timestamp)')
        cursor.execute('CREATE INDEX idx_interactions_dataset_split ON interactions(dataset_split)')
        cursor.execute('CREATE INDEX idx_users_user_id ON users(user_id)')
        logger.info("Created indexes")
        
        conn.commit()
        
        # Import dữ liệu users
        logger.info("Importing users data...")
        user_file = os.path.join(data_dir, "user.csv")
        if os.path.exists(user_file):
            df_users = pd.read_csv(user_file)
            # Đảm bảo có đủ columns
            required_cols = ['user_id', 'age', 'gender', 'occupation', 'user_profile']
            for col in required_cols:
                if col not in df_users.columns:
                    df_users[col] = None
            
            df_users = df_users[required_cols]
            # Xử lý giá trị null
            df_users['age'] = df_users['age'].fillna(0)
            df_users['gender'] = df_users['gender'].fillna('unknown')
            df_users['occupation'] = df_users['occupation'].fillna('unknown')
            df_users['user_profile'] = df_users['user_profile'].fillna('')
            
            df_users.to_sql('users', conn, if_exists='append', index=False)
            logger.info(f"Imported {len(df_users)} users")
        else:
            logger.warning(f"User file not found: {user_file}")
        
        # Import dữ liệu items
        logger.info("Importing items data...")
        item_file = os.path.join(data_dir, "item.csv")
        if os.path.exists(item_file):
            df_items = pd.read_csv(item_file)
            # Đảm bảo có đủ columns
            required_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'genre', 'item_attributes']
            for col in required_cols:
                if col not in df_items.columns:
                    df_items[col] = None
            
            df_items = df_items[required_cols]
            # Xử lý giá trị null
            df_items['title'] = df_items['title'].fillna('Unknown')
            df_items['release_date'] = df_items['release_date'].fillna('unknown')
            df_items['video_release_date'] = df_items['video_release_date'].fillna('unknown')
            df_items['genre'] = df_items['genre'].fillna('Unknown')
            df_items['item_attributes'] = df_items['item_attributes'].fillna('')
            
            df_items.to_sql('items', conn, if_exists='append', index=False)
            logger.info(f"Imported {len(df_items)} items")
        else:
            logger.warning(f"Item file not found: {item_file}")
        
        # Import dữ liệu interactions từ train, dev, test
        logger.info("Importing interactions data...")
        interaction_files = [
            ("train.csv", "train"),
            ("dev.csv", "dev"),
            ("test.csv", "test")
        ]
        
        total_interactions = 0
        for filename, split_name in interaction_files:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                logger.info(f"Reading {filename}...")
                try:
                    # Đọc từng chunk để tránh memory error với file lớn
                    chunk_size = 10000
                    chunks_processed = 0
                    rows_imported = 0
                    
                    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
                        # Kiểm tra columns
                        required_cols = ['user_id', 'item_id', 'rating']
                        if not all(col in chunk.columns for col in required_cols):
                            logger.warning(f"Missing required columns in {filename}. Expected: {required_cols}, Got: {chunk.columns.tolist()}")
                            break
                        
                        # Thêm dataset_split
                        chunk['dataset_split'] = split_name
                        
                        # Xử lý các columns optional - lấy tất cả columns có sẵn
                        optional_cols = ['timestamp', 'neg_item_id', 'position', 'history_item_id', 
                                        'history_rating', 'history', 'user_profile', 'target_item_attributes',
                                        'candidate_item_id', 'candidate_item_attributes']
                        for col in optional_cols:
                            if col not in chunk.columns:
                                chunk[col] = None
                            else:
                                # Convert NaN to None
                                chunk[col] = chunk[col].where(pd.notna(chunk[col]), None)
                        
                        # Chọn và sắp xếp columns
                        all_cols = ['user_id', 'item_id', 'rating', 'timestamp', 'dataset_split'] + optional_cols
                        chunk = chunk[[col for col in all_cols if col in chunk.columns]]
                        
                        # Đảm bảo các giá trị NaN được convert thành None cho SQLite
                        chunk = chunk.where(pd.notna(chunk), None)
                        
                        # Import chunk
                        chunk.to_sql('interactions', conn, if_exists='append', index=False)
                        chunks_processed += 1
                        rows_imported += len(chunk)
                        
                        if chunks_processed % 10 == 0:
                            logger.info(f"  Processed {chunks_processed} chunks, {rows_imported} rows...")
                    
                    logger.info(f"Imported {rows_imported} interactions from {filename}")
                    total_interactions += rows_imported
                    
                except Exception as e:
                    logger.error(f"Error importing {filename}: {e}")
            else:
                logger.warning(f"File not found: {filepath}")
    
    else:  # Beauty dataset
        # Tạo bảng items cho Beauty
        cursor.execute('''
            CREATE TABLE items (
                item_id INTEGER PRIMARY KEY,
                title TEXT,
                brand TEXT,
                price REAL,
                categories TEXT,
                item_attributes TEXT
            )
        ''')
        logger.info("Created table: items")
        
        # Tạo bảng interactions cho Beauty
        cursor.execute('''
            CREATE TABLE interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                item_id INTEGER NOT NULL,
                rating REAL NOT NULL,
                summary TEXT,
                timestamp INTEGER,
                dataset_split TEXT,
                FOREIGN KEY (item_id) REFERENCES items(item_id)
            )
        ''')
        logger.info("Created table: interactions")
        
        # Tạo index để tăng tốc query
        cursor.execute('CREATE INDEX idx_interactions_user_id ON interactions(user_id)')
        cursor.execute('CREATE INDEX idx_interactions_item_id ON interactions(item_id)')
        cursor.execute('CREATE INDEX idx_interactions_rating ON interactions(rating)')
        cursor.execute('CREATE INDEX idx_interactions_timestamp ON interactions(timestamp)')
        logger.info("Created indexes")
        
        conn.commit()
        
        # Import dữ liệu items
        logger.info("Importing items data...")
        item_file = os.path.join(data_dir, "item.csv")
        if os.path.exists(item_file):
            df_items = pd.read_csv(item_file)
            # Đảm bảo có đủ columns
            required_cols = ['item_id', 'title', 'brand', 'price', 'categories', 'item_attributes']
            for col in required_cols:
                if col not in df_items.columns:
                    if col == 'item_attributes':
                        # Tạo item_attributes từ các trường khác nếu không có
                        df_items['item_attributes'] = df_items.apply(
                            lambda x: f"Brand: {x.get('brand', 'unknown')}, Price: {x.get('price', 'unknown')}, Categories: {x.get('categories', 'unknown')}",
                            axis=1
                        )
                    else:
                        df_items[col] = None
            
            df_items = df_items[required_cols]
            # Xử lý giá trị null - sử dụng fillna với value cụ thể
            df_items['title'] = df_items['title'].fillna('Unknown')
            df_items['brand'] = df_items['brand'].fillna('unknown')
            df_items['categories'] = df_items['categories'].fillna('Unknown')
            df_items['item_attributes'] = df_items['item_attributes'].fillna('')
            
            # Convert price nếu là string
            if df_items['price'].dtype == 'object':
                df_items['price'] = pd.to_numeric(df_items['price'], errors='coerce')
            df_items['price'] = df_items['price'].fillna(0.0)
            
            df_items.to_sql('items', conn, if_exists='append', index=False)
            logger.info(f"Imported {len(df_items)} items")
        else:
            logger.warning(f"Item file not found: {item_file}")
        
        # Import dữ liệu interactions từ train, dev, test
        logger.info("Importing interactions data...")
        interaction_files = [
            ("train.csv", "train"),
            ("dev.csv", "dev"),
            ("test.csv", "test")
        ]
        
        total_interactions = 0
        for filename, split_name in interaction_files:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                logger.info(f"Reading {filename}...")
                try:
                    # Đọc từng chunk để tránh memory error với file lớn
                    chunk_size = 10000
                    chunks_processed = 0
                    rows_imported = 0
                    
                    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
                        # Kiểm tra columns
                        required_cols = ['user_id', 'item_id', 'rating']
                        if not all(col in chunk.columns for col in required_cols):
                            logger.warning(f"Missing required columns in {filename}. Expected: {required_cols}, Got: {chunk.columns.tolist()}")
                            break
                        
                        # Thêm dataset_split
                        chunk['dataset_split'] = split_name
                        
                        # Xử lý các columns optional
                        if 'summary' not in chunk.columns:
                            chunk['summary'] = None
                        else:
                            # Convert NaN to None for summary
                            chunk['summary'] = chunk['summary'].where(pd.notna(chunk['summary']), None)
                        
                        if 'timestamp' not in chunk.columns:
                            chunk['timestamp'] = None
                        else:
                            # Convert NaN to None for timestamp
                            chunk['timestamp'] = chunk['timestamp'].where(pd.notna(chunk['timestamp']), None)
                        
                        # Chọn và sắp xếp columns
                        chunk = chunk[['user_id', 'item_id', 'rating', 'summary', 'timestamp', 'dataset_split']]
                        
                        # Đảm bảo các giá trị NaN được convert thành None cho SQLite
                        chunk = chunk.where(pd.notna(chunk), None)
                        
                        # Import chunk
                        chunk.to_sql('interactions', conn, if_exists='append', index=False)
                        chunks_processed += 1
                        rows_imported += len(chunk)
                        
                        if chunks_processed % 10 == 0:
                            logger.info(f"  Processed {chunks_processed} chunks, {rows_imported} rows...")
                    
                    logger.info(f"Imported {rows_imported} interactions from {filename}")
                    total_interactions += rows_imported
                    
                except Exception as e:
                    logger.error(f"Error importing {filename}: {e}")
            else:
                logger.warning(f"File not found: {filepath}")
        
        # Tạo bảng users (tổng hợp từ interactions) cho Beauty
        logger.info("Creating users table...")
        cursor.execute('''
            CREATE TABLE users AS
            SELECT DISTINCT 
                user_id,
                COUNT(DISTINCT item_id) as total_items,
                AVG(rating) as avg_rating,
                MIN(timestamp) as first_interaction,
                MAX(timestamp) as last_interaction
            FROM interactions
            GROUP BY user_id
        ''')
        cursor.execute('CREATE INDEX idx_users_user_id ON users(user_id)')
        logger.info("Created table: users")
    
    # Thống kê
    cursor.execute('SELECT COUNT(*) FROM items')
    num_items = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM interactions')
    num_interactions = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM users')
    num_users = cursor.fetchone()[0]
    
    cursor.execute('SELECT AVG(rating) FROM interactions WHERE rating IS NOT NULL')
    avg_rating = cursor.fetchone()[0]
    
    logger.info("\n" + "="*50)
    logger.info("Database created successfully!")
    logger.info(f"Database path: {db_path}")
    logger.info(f"Number of items: {num_items:,}")
    logger.info(f"Number of users: {num_users:,}")
    logger.info(f"Number of interactions: {num_interactions:,}")
    logger.info(f"Average rating: {avg_rating:.2f}" if avg_rating else "Average rating: N/A")
    logger.info("="*50)
    
    conn.commit()
    conn.close()
    
    return db_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create SQLite database from supported datasets")
    parser.add_argument("--dataset", type=str, default="ml-100k", choices=list(DEFAULT_DATASET_CONFIG.keys()),
                       help="Dataset to import (default: ml-100k)")
    parser.add_argument("--db_path", type=str, default=None,
                       help="Optional override for SQLite database file path")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Optional override for directory containing CSV files")
    
    args = parser.parse_args()
    
    create_database(db_path=args.db_path, data_dir=args.data_dir, dataset=args.dataset)

