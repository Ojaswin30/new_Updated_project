import gzip
import json
import sqlite3
from pathlib import Path
from typing import List

from ml.src.sentiment.sentiment_inference import classify_sentiment_batch

# -----------------------
# PATHS (DO NOT GUESS)
# -----------------------

ROOT = Path(__file__).resolve().parents[3]

REVIEWS_DIR = ROOT / "ml" / "data" / "reviews"
OUTPUT_DB = ROOT / "ml" / "data" / "reviews-output" / "reviews_enriched.sqlite"

BATCH_SIZE = 256

# -----------------------
# DB SETUP
# -----------------------

conn = sqlite3.connect(OUTPUT_DB)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    product_id TEXT,
    brand TEXT,
    category TEXT,
    rating REAL,
    sentiment TEXT,
    review_text TEXT
)
""")

conn.commit()

# -----------------------
# INGESTION
# -----------------------

def ingest_file(gz_path: Path):
    texts: List[str] = []
    buffer: List[dict] = []

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)

            text = r.get("reviewText") or ""
            if not text:
                continue

            buffer.append({
                "review_id": r.get("review_id"),
                "product_id": r.get("parent_asin"),
                "brand": r.get("brand"),
                "category": r.get("category"),
                "rating": float(r.get("rating", 0.0)),
                "text": text,
            })

            texts.append(text)

            if len(buffer) >= BATCH_SIZE:
                _flush(buffer, texts)

    if buffer:
        _flush(buffer, texts)


def _flush(buffer: List[dict], texts: List[str]):
    sentiments = classify_sentiment_batch(texts)

    for r, s in zip(buffer, sentiments):
        cur.execute(
            """
            INSERT OR IGNORE INTO reviews
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["review_id"],
                r["product_id"],
                r["brand"],
                r["category"],
                r["rating"],
                s,
                r["text"],
            ),
        )

    conn.commit()
    buffer.clear()
    texts.clear()


# -----------------------
# ENTRY POINT
# -----------------------

if __name__ == "__main__":
    for gz in REVIEWS_DIR.glob("*.jsonl.gz"):
        print(f"Ingesting {gz.name}")
        ingest_file(gz)

    conn.close()
    print("DONE")
