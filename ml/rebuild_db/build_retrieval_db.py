"""
Fast data pipeline for intent-aware multimodal retrieval evaluation
Generates synthetic queries + builds SQLite DB for Amazon Fashion
"""
import gzip
import json
import sqlite3
import random
from pathlib import Path
from typing import List, Dict, Tuple
import re

# ==================== CONFIG ====================
CATEGORY = "Amazon_Fashion"
DATA_DIR = Path("data/reviews")  # Adjust to your path
META_FILE = DATA_DIR / f"meta_{CATEGORY}.jsonl.gz"
REVIEW_FILE = DATA_DIR / f"{CATEGORY}.jsonl.gz"
OUTPUT_DB = Path("data/reviews-output/product_ranking.sqlite")

# Limits for speed
MAX_PRODUCTS = 1000  # Sample 1000 products for fast experimentation
MIN_REVIEWS_PER_PRODUCT = 3  # Products need at least 3 reviews
QUERIES_PER_PRODUCT = 3  # Generate 3 synthetic queries per product

# ==================== QUERY GENERATION ====================

# Extract attributes from product title
def extract_attributes(title: str) -> Dict[str, str]:
    """Extract color, material, size hints from title"""
    title_lower = title.lower()
    
    # Common colors
    colors = ['black', 'white', 'blue', 'red', 'green', 'pink', 'gray', 'grey', 
              'brown', 'navy', 'beige', 'gold', 'silver', 'purple', 'yellow', 'orange']
    color = next((c for c in colors if c in title_lower), None)
    
    # Common materials
    materials = ['cotton', 'leather', 'denim', 'silk', 'wool', 'polyester', 
                 'canvas', 'suede', 'linen', 'nylon']
    material = next((m for m in materials if m in title_lower), None)
    
    # Size hints
    sizes = ['small', 'medium', 'large', 'xl', 'xxl', 's', 'm', 'l']
    size = next((s for s in sizes if f' {s} ' in f' {title_lower} ' or f' {s},' in f' {title_lower} '), None)
    
    # Product type (first few words usually indicate category)
    words = title.split()
    product_type = ' '.join(words[:3]) if len(words) >= 3 else title
    
    return {
        'color': color,
        'material': material,
        'size': size,
        'product_type': product_type
    }

def generate_queries(product: Dict) -> List[Tuple[str, str]]:
    """Generate synthetic queries: (query_text, intent_type)"""
    title = product['title']
    attrs = extract_attributes(title)
    queries = []
    
    # Visual-dominant query (color + product type)
    if attrs['color']:
        q = f"{attrs['color']} {attrs['product_type'].split()[0]}"
        queries.append((q, 'visual'))
    
    # Attribute-dominant query (material + size + type)
    parts = []
    if attrs['material']:
        parts.append(attrs['material'])
    if attrs['size']:
        parts.append(attrs['size'])
    if parts:
        parts.append(attrs['product_type'].split()[0])
        queries.append((' '.join(parts), 'attribute'))
    
    # Hybrid query (color + material + type)
    parts = []
    if attrs['color']:
        parts.append(attrs['color'])
    if attrs['material']:
        parts.append(attrs['material'])
    if parts:
        parts.append(attrs['product_type'].split()[0])
        queries.append((' '.join(parts), 'hybrid'))
    
    # Fallback: just product type
    if not queries:
        queries.append((attrs['product_type'], 'attribute'))
    
    return queries[:QUERIES_PER_PRODUCT]

# ==================== DATA LOADING ====================

def load_products(limit: int = MAX_PRODUCTS) -> Dict[str, Dict]:
    """Load product metadata"""
    print(f"Loading products from {META_FILE}...")
    products = {}
    
    with gzip.open(META_FILE, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            data = json.loads(line)
            
            # Must have image and valid title
            if not data.get('images') or not data.get('title'):
                continue
            
            parent_asin = data['parent_asin']
            products[parent_asin] = {
                'parent_asin': parent_asin,
                'title': data['title'],
                'category': data.get('main_category', 'Unknown'),
                'image_url': data['images'][0]['hi_res'] if data['images'] else None,
                'store': data.get('store', 'Unknown'),
                'average_rating': data.get('average_rating', 0.0),
                'rating_number': data.get('rating_number', 0),
            }
    
    print(f"Loaded {len(products)} products")
    return products

def load_reviews(products: Dict[str, Dict]) -> Dict[str, List[Dict]]:
    """Load reviews, group by parent_asin"""
    print(f"Loading reviews from {REVIEW_FILE}...")
    reviews_by_product = {asin: [] for asin in products}
    
    with gzip.open(REVIEW_FILE, 'rt', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            parent_asin = data.get('parent_asin')
            
            if parent_asin in products:
                reviews_by_product[parent_asin].append({
                    'rating': data['rating'],
                    'text': data.get('text', ''),
                    'verified': data.get('verified_purchase', False)
                })
    
    # Filter: only keep products with enough reviews
    filtered = {k: v for k, v in reviews_by_product.items() 
                if len(v) >= MIN_REVIEWS_PER_PRODUCT}
    
    print(f"Found {len(filtered)} products with >={MIN_REVIEWS_PER_PRODUCT} reviews")
    return filtered

# ==================== DATABASE BUILD ====================

def build_database(products: Dict, reviews: Dict):
    """Build SQLite database with products, reviews, queries, ground truth"""
    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    
    if OUTPUT_DB.exists():
        OUTPUT_DB.unlink()
    
    conn = sqlite3.connect(OUTPUT_DB)
    c = conn.cursor()
    
    # Products table
    c.execute('''
        CREATE TABLE products (
            parent_asin TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            category TEXT,
            image_url TEXT,
            store TEXT,
            average_rating REAL,
            rating_number INTEGER
        )
    ''')
    
    # Reviews table
    c.execute('''
        CREATE TABLE reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_asin TEXT NOT NULL,
            rating REAL NOT NULL,
            text TEXT,
            verified INTEGER,
            FOREIGN KEY (parent_asin) REFERENCES products(parent_asin)
        )
    ''')
    
    # Product ranking (aggregated review scores)
    c.execute('''
        CREATE TABLE product_ranking (
            parent_asin TEXT PRIMARY KEY,
            review_score REAL NOT NULL,
            num_reviews INTEGER NOT NULL,
            avg_rating REAL NOT NULL,
            FOREIGN KEY (parent_asin) REFERENCES products(parent_asin)
        )
    ''')
    
    # Synthetic queries with ground truth
    c.execute('''
        CREATE TABLE queries (
            query_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT NOT NULL,
            intent_type TEXT NOT NULL,
            relevant_asin TEXT NOT NULL,
            FOREIGN KEY (relevant_asin) REFERENCES products(parent_asin)
        )
    ''')
    
    print("Inserting products...")
    for asin, prod in products.items():
        if asin not in reviews:
            continue
        c.execute('''
            INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (asin, prod['title'], prod['category'], prod['image_url'],
              prod['store'], prod['average_rating'], prod['rating_number']))
    
    print("Inserting reviews...")
    for asin, rev_list in reviews.items():
        for rev in rev_list:
            c.execute('''
                INSERT INTO reviews (parent_asin, rating, text, verified)
                VALUES (?, ?, ?, ?)
            ''', (asin, rev['rating'], rev['text'], 1 if rev['verified'] else 0))
    
    print("Computing product ranking scores...")
    for asin, rev_list in reviews.items():
        ratings = [r['rating'] for r in rev_list]
        avg_rating = sum(ratings) / len(ratings)
        num_reviews = len(ratings)
        
        # Review score: normalized avg rating, boosted by log(num_reviews)
        import math
        review_score = (avg_rating / 5.0) * (1 + 0.1 * math.log(1 + num_reviews))
        review_score = min(1.0, review_score)  # Cap at 1.0
        
        c.execute('''
            INSERT INTO product_ranking VALUES (?, ?, ?, ?)
        ''', (asin, review_score, num_reviews, avg_rating))
    
    print("Generating synthetic queries...")
    for asin, prod in products.items():
        if asin not in reviews:
            continue
        queries = generate_queries(prod)
        for query_text, intent_type in queries:
            c.execute('''
                INSERT INTO queries (query_text, intent_type, relevant_asin)
                VALUES (?, ?, ?)
            ''', (query_text, intent_type, asin))
    
    conn.commit()
    
    # Print stats
    c.execute("SELECT COUNT(*) FROM products")
    print(f"\nDatabase built: {OUTPUT_DB}")
    print(f"  Products: {c.fetchone()[0]}")
    c.execute("SELECT COUNT(*) FROM reviews")
    print(f"  Reviews: {c.fetchone()[0]}")
    c.execute("SELECT COUNT(*) FROM queries")
    print(f"  Queries: {c.fetchone()[0]}")
    c.execute("SELECT intent_type, COUNT(*) FROM queries GROUP BY intent_type")
    print("  Query breakdown:")
    for intent, count in c.fetchall():
        print(f"    {intent}: {count}")
    
    conn.close()

# ==================== MAIN ====================

if __name__ == "__main__":
    print("="*60)
    print("BUILDING RETRIEVAL EVALUATION DATABASE")
    print("="*60)
    
    products = load_products(MAX_PRODUCTS)
    reviews = load_reviews(products)
    
    # Only keep products that have reviews
    products = {k: v for k, v in products.items() if k in reviews}
    
    print(f"\nFinal dataset: {len(products)} products with reviews")
    
    build_database(products, reviews)
    
    print("\n✓ Done! Database ready for evaluation.")
    print(f"  Location: {OUTPUT_DB}")
