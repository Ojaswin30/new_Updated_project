import gzip
import json
import sqlite3
from pathlib import Path
from typing import List

from ml.src.sentiment.sentiment_inference import classify_sentiment_batch

# =====================================================
# PATHS (MATCH YOUR REPO)
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

REVIEWS_DIR = ROOT / "ml" / "data" / "reviews"
FINAL_DB = ROOT / "ml" / "data" / "reviews-output" / "reviews_final.sqlite"

BATCH_SIZE = 256

# =====================================================
# DB SETUP
# =====================================================

conn = sqlite3.connect(FINAL_DB)
cur = conn.cursor()

# Raw reviews (optional but useful)
cur.execute("""
CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    product_id TEXT,
    brand TEXT,
    category TEXT,
    rating REAL,
    sentiment TEXT
)
""")

# Product-level aggregation
cur.execute("""
CREATE TABLE IF NOT EXISTS product_scores (
    product_id TEXT PRIMARY KEY,
    brand TEXT,
    category TEXT,
    avg_rating REAL,
    avg_sentiment REAL,
    review_count INTEGER
)
""")

# Brand × Category aggregation (FINAL GOAL)
cur.execute("""
CREATE TABLE IF NOT EXISTS brand_category_scores (
    brand TEXT,
    category TEXT,
    avg_rating REAL,
    avg_sentiment REAL,
    review_count INTEGER,
    weighted_score REAL,
    PRIMARY KEY (brand, category)
)
""")

conn.commit()

# =====================================================
# INGEST + SENTIMENT
# =====================================================

def flush(buffer: List[dict], texts: List[str]):
    sentiments = classify_sentiment_batch(texts)

    for r, s in zip(buffer, sentiments):
        cur.execute(
            """
            INSERT OR IGNORE INTO reviews
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                r["review_id"],
                r["product_id"],
                r["brand"],
                r["category"],
                r["rating"],
                1 if s == "positive" else 0,
            ),
        )

    conn.commit()
    buffer.clear()
    texts.clear()


buffer, texts = [], []

for gz in REVIEWS_DIR.glob("*.jsonl.gz"):
    print(f"Processing {gz.name}")

    with gzip.open(gz, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)

            if not r.get("reviewText"):
                continue

            buffer.append({
                "review_id": r.get("review_id"),
                "product_id": r.get("parent_asin"),
                "brand": r.get("brand"),
                "category": r.get("category"),
                "rating": float(r.get("rating", 0.0)),
            })
            texts.append(r["reviewText"])

            if len(buffer) >= BATCH_SIZE:
                flush(buffer, texts)

if buffer:
    flush(buffer, texts)

# =====================================================
# PRODUCT-LEVEL AGGREGATION
# =====================================================

cur.execute("""
INSERT OR REPLACE INTO product_scores
SELECT
    product_id,
    brand,
    category,
    AVG(rating) AS avg_rating,
    AVG(sentiment) AS avg_sentiment,
    COUNT(*) AS review_count
FROM reviews
GROUP BY product_id
""")

# =====================================================
# BRAND × CATEGORY AGGREGATION (FINAL INTELLIGENCE)
# =====================================================

cur.execute("""
INSERT OR REPLACE INTO brand_category_scores
SELECT
    brand,
    category,
    AVG(avg_rating) AS avg_rating,
    AVG(avg_sentiment) AS avg_sentiment,
    SUM(review_count) AS review_count,
    AVG(avg_rating) * LOG(SUM(review_count) + 1) AS weighted_score
FROM product_scores
WHERE brand IS NOT NULL AND category IS NOT NULL
GROUP BY brand, category
""")

conn.commit()
conn.close()

print("FINAL DATABASE BUILT SUCCESSFULLY")
