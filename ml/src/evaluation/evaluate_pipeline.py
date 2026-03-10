# ml/src/evaluation/evaluate_pipeline.py

import sqlite3
import json
import time
from typing import List, Dict, Tuple
from pathlib import Path

# ==================== CONFIG ====================
DB_PATH     = "D:\\github\\git repositories\\new_Updated_project\\ml\\data\\reviews-output\\product_ranking1.sqlite"
TOP_K_LIST  = [5, 10, 20]      # evaluate at multiple cutoffs
MAX_QUERIES = 500               # set to None to run all 85K (slow)
INTENT_FILTER = None            # set to 'visual', 'hybrid', 'attribute' to test one type

# ==================== LOAD GROUND TRUTH ====================
def load_ground_truth(db_path: str, limit=None, intent_filter=None) -> List[Dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c    = conn.cursor()

    query = "SELECT query_id, query_text, intent_type, relevant_asin FROM queries"
    if intent_filter:
        query += f" WHERE intent_type = '{intent_filter}'"
    if limit:
        query += f" LIMIT {limit}"

    c.execute(query)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    print(f"Loaded {len(rows)} ground truth queries")
    return rows

# ==================== SQL-ONLY RETRIEVAL (no image) ====================
def retrieve_by_text(query_text: str, db_path: str, top_k: int = 20) -> List[str]:
    """
    Text-only retrieval using keyword search + review score ranking.
    Used for evaluation since we don't have images for each query.
    Returns list of parent_asin in ranked order.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c    = conn.cursor()

    # Split query into keywords
    keywords = query_text.lower().strip().split()

    where_clauses = ["1=1"]
    params        = {}

    # Keyword match on title
    for i, kw in enumerate(keywords[:5]):
        key = f"kw{i}"
        where_clauses.append(f"(LOWER(p.title) LIKE :{key} OR LOWER(p.features) LIKE :{key})")
        params[key] = f"%{kw}%"

    params["limit"] = top_k

    sql = f"""
        SELECT p.parent_asin,
               COALESCE(r.review_score, 0.0) AS review_score
        FROM products p
        LEFT JOIN product_ranking r ON p.parent_asin = r.parent_asin
        WHERE {' AND '.join(where_clauses)}
        ORDER BY review_score DESC, p.average_rating DESC
        LIMIT :limit
    """

    c.execute(sql, params)
    results = [row["parent_asin"] for row in c.fetchall()]
    conn.close()
    return results

# ==================== METRICS ====================
def precision_at_k(retrieved: List[str], relevant: str, k: int) -> float:
    """Precision@K — how many of top-K are relevant"""
    top_k = retrieved[:k]
    return 1.0 if relevant in top_k else 0.0

def recall_at_k(retrieved: List[str], relevant: str, k: int) -> float:
    """Recall@K — since we have 1 relevant item, same as hit rate"""
    return precision_at_k(retrieved, relevant, k)

def reciprocal_rank(retrieved: List[str], relevant: str) -> float:
    """MRR — 1/rank of first relevant result"""
    for i, asin in enumerate(retrieved):
        if asin == relevant:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(retrieved: List[str], relevant: str, k: int) -> float:
    """NDCG@K — with single relevant item, DCG = 1/log2(rank+1) if found"""
    top_k = retrieved[:k]
    for i, asin in enumerate(top_k):
        if asin == relevant:
            import math
            return 1.0 / math.log2(i + 2)   # i+2 because log2(1) = 0
    return 0.0

# ==================== MAIN EVALUATION ====================
def evaluate(db_path: str, limit=None, intent_filter=None):
    ground_truth = load_ground_truth(db_path, limit, intent_filter)

    max_k        = max(TOP_K_LIST)
    results_log  = []

    # Accumulators per K
    metrics = {
        k: {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}
        for k in TOP_K_LIST
    }
    mrr_total    = 0.0
    total        = 0
    zero_results = 0
    start        = time.time()

    for i, gt in enumerate(ground_truth):
        query_text    = gt["query_text"]
        relevant_asin = gt["relevant_asin"]
        intent_type   = gt["intent_type"]

        retrieved = retrieve_by_text(query_text, db_path, top_k=max_k)

        if not retrieved:
            zero_results += 1
            continue

        total += 1

        # Compute metrics at each K
        for k in TOP_K_LIST:
            metrics[k]["precision"] += precision_at_k(retrieved, relevant_asin, k)
            metrics[k]["recall"]    += recall_at_k(retrieved, relevant_asin, k)
            metrics[k]["ndcg"]      += ndcg_at_k(retrieved, relevant_asin, k)

        mrr_total += reciprocal_rank(retrieved, relevant_asin)

        # Log individual result
        rank = None
        for j, asin in enumerate(retrieved):
            if asin == relevant_asin:
                rank = j + 1
                break

        results_log.append({
            "query_id":      gt["query_id"],
            "query_text":    query_text,
            "intent_type":   intent_type,
            "relevant_asin": relevant_asin,
            "rank":          rank,
            "found_in_top20": rank is not None and rank <= 20,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(ground_truth)}] elapsed: {elapsed:.1f}s")

    # ==================== PRINT RESULTS ====================
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Queries evaluated : {total}")
    print(f"Zero-result queries: {zero_results}")
    print(f"MRR               : {mrr_total / total:.4f}")
    print()

    for k in TOP_K_LIST:
        p  = metrics[k]["precision"] / total
        r  = metrics[k]["recall"]    / total
        nd = metrics[k]["ndcg"]      / total
        print(f"@{k:2d}  Precision: {p:.4f}  |  Recall: {r:.4f}  |  NDCG: {nd:.4f}")

    # Breakdown by intent type
    print("\n--- Breakdown by Intent Type ---")
    for intent in ["visual", "hybrid", "attribute"]:
        subset = [r for r in results_log if r["intent_type"] == intent]
        if not subset:
            continue
        hit = sum(1 for r in subset if r["found_in_top20"])
        print(f"  {intent:10s}: {hit}/{len(subset)} found in top-20  ({100*hit/len(subset):.1f}%)")

    # Top failing queries
    print("\n--- Sample Failed Queries (not found in top-20) ---")
    failed = [r for r in results_log if not r["found_in_top20"]][:10]
    for f in failed:
        print(f"  Query: '{f['query_text']}' | Expected: {f['relevant_asin']}")

    # Save full log
    log_path = "data/reviews-output/eval_results.json"
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({
            "total_queries": total,
            "zero_results":  zero_results,
            "mrr":           mrr_total / total,
            "metrics_by_k":  {
                str(k): {
                    "precision": metrics[k]["precision"] / total,
                    "recall":    metrics[k]["recall"]    / total,
                    "ndcg":      metrics[k]["ndcg"]      / total,
                }
                for k in TOP_K_LIST
            },
            "per_query_log": results_log,
        }, f, indent=2)

    print(f"\nFull log saved to: {log_path}")
    return results_log


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",     default=DB_PATH)
    parser.add_argument("--limit",  type=int, default=MAX_QUERIES)
    parser.add_argument("--intent", default=INTENT_FILTER,
                        choices=["visual", "hybrid", "attribute", None])
    args = parser.parse_args()

    evaluate(
        db_path       = args.db,
        limit         = args.limit,
        intent_filter = args.intent,
    )