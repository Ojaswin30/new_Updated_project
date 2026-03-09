"""
Evaluation with TEXT-DERIVED visual importance scores
No actual image processing needed - infer visual dependency from query text
"""
import sqlite3
import numpy as np
from typing import List, Dict
from collections import defaultdict
import argparse

from ml.src.pipeline.constraint_parser import ConstraintParser
from ml.training.intent_classifier import IntentClassifier


# ==================== METRICS ====================

def dcg_at_k(relevance_scores: List[float], k: int) -> float:
    relevance_scores = relevance_scores[:k]
    if not relevance_scores:
        return 0.0
    gains = [(2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevance_scores)]
    return sum(gains)

def ndcg_at_k(relevance_scores: List[float], k: int) -> float:
    dcg = dcg_at_k(relevance_scores, k)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal_scores, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg

def precision_at_k(relevance_scores: List[float], k: int, threshold: float = 0.5) -> float:
    relevance_scores = relevance_scores[:k]
    if not relevance_scores:
        return 0.0
    relevant = sum(1 for score in relevance_scores if score >= threshold)
    return relevant / len(relevance_scores)

def mrr(relevance_scores: List[float], threshold: float = 0.5) -> float:
    for i, score in enumerate(relevance_scores, 1):
        if score >= threshold:
            return 1.0 / i
    return 0.0


# ==================== VISUAL IMPORTANCE FROM TEXT ====================

def compute_visual_importance(constraints: Dict) -> float:
    """
    Compute how visually-dependent a query is from parsed text alone
    
    High visual importance (0.7-0.9):
    - Has color
    - Few other constraints
    
    Low visual importance (0.1-0.3):
    - Has size/price/material
    - Many specific constraints
    
    Returns: visual importance score [0.0, 1.0]
    """
    score = 0.5  # baseline
    
    # Visual indicators
    if constraints.get('color'):
        score += 0.3
    
    if constraints.get('category'):
        score += 0.1
    
    # Non-visual indicators (reduce visual importance)
    if constraints.get('size'):
        score -= 0.2
    
    if constraints.get('material'):
        score -= 0.15
    
    if constraints.get('price_min') or constraints.get('price_max'):
        score -= 0.2
    
    keywords = constraints.get('keywords', [])
    if len(keywords) >= 3:
        score -= 0.15  # Many keywords = specific search, less visual
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


def compute_product_visual_match(query_constraints: Dict, product_title: str) -> float:
    """
    How well does product match visual aspects of query?
    Based on text matching of visual features
    """
    score = 0.5  # baseline
    title_lower = product_title.lower()
    
    # Color match
    query_color = query_constraints.get('color')
    if query_color:
        if query_color.lower() in title_lower:
            score += 0.4  # Strong color match
        else:
            score -= 0.3  # Color mismatch hurts
    
    # Category match
    query_category = query_constraints.get('category')
    if query_category:
        if query_category.lower() in title_lower:
            score += 0.2
    
    return max(0.0, min(1.0, score))


# ==================== RETRIEVAL ====================

def word_overlap_score(query: str, title: str) -> float:
    query_words = set(query.lower().split())
    title_words = set(title.lower().split())
    if not query_words:
        return 0.0
    overlap = len(query_words & title_words)
    return overlap / len(query_words)


def retrieve_and_score(query_text: str, db_path: str, parser: ConstraintParser, 
                       top_k: int = 200) -> List[Dict]:
    """Retrieve products and compute text, visual, review scores"""
    
    # Parse query
    constraints = parser.parse(query_text)
    visual_importance = compute_visual_importance(constraints)
    
    # Get products
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("""
        SELECT p.parent_asin, p.title, p.category, p.average_rating,
               COALESCE(r.review_score, 0.0) AS review_score
        FROM products p
        LEFT JOIN product_ranking r ON p.parent_asin = r.parent_asin
    """)
    
    all_products = [dict(row) for row in c.fetchall()]
    conn.close()
    
    # Score each product
    scored = []
    for product in all_products:
        # Text score (word overlap)
        text_score = word_overlap_score(query_text, product['title'])
        if text_score == 0:
            continue
        
        # Visual score (text-based matching of visual features)
        visual_score = compute_product_visual_match(constraints, product['title'])
        
        product['text_score'] = text_score
        product['visual_score'] = visual_score
        product['visual_importance'] = visual_importance  # How much to trust visual
        scored.append(product)
    
    scored.sort(key=lambda x: x['text_score'], reverse=True)
    return scored[:top_k]


# ==================== FUSION ====================

def static_fusion(products: List[Dict]) -> List[Dict]:
    """Static fusion: α=0.50, β=0.35, γ=0.15"""
    alpha, beta, gamma = 0.50, 0.35, 0.15
    
    ranked = []
    for p in products:
        final_score = (
            alpha * p['visual_score'] +
            beta * p['text_score'] +
            gamma * p['review_score']
        )
        p['final_score'] = final_score
        p['fusion_weights'] = {'alpha': alpha, 'beta': beta, 'gamma': gamma}
        ranked.append(p)
    
    ranked.sort(key=lambda x: x['final_score'], reverse=True)
    return ranked


def dynamic_fusion(products: List[Dict], intent: str) -> List[Dict]:
    """Dynamic fusion based on detected intent"""
    
    # Adjust weights based on intent
    if intent == 'visual':
        alpha, beta, gamma = 0.70, 0.15, 0.15
    elif intent == 'attribute':
        alpha, beta, gamma = 0.20, 0.60, 0.20
    else:  # hybrid
        alpha, beta, gamma = 0.45, 0.40, 0.15
    
    ranked = []
    for p in products:
        final_score = (
            alpha * p['visual_score'] +
            beta * p['text_score'] +
            gamma * p['review_score']
        )
        p['final_score'] = final_score
        p['fusion_weights'] = {'alpha': alpha, 'beta': beta, 'gamma': gamma}
        ranked.append(p)
    
    ranked.sort(key=lambda x: x['final_score'], reverse=True)
    return ranked


# ==================== EVALUATION ====================

class TextBasedEvaluator:
    def __init__(self, db_path: str, k_values: List[int] = [5, 10, 20]):
        self.db_path = db_path
        self.k_values = k_values
        self.parser = ConstraintParser()
        self.classifier = IntentClassifier()
        
        self.queries = self._load_queries()
        print(f"Loaded {len(self.queries)} evaluation queries")
    
    def _load_queries(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM queries")
        queries = [dict(row) for row in c.fetchall()]
        conn.close()
        return queries
    
    def evaluate(self, use_dynamic: bool = False) -> Dict:
        results_by_intent = defaultdict(list)
        all_results = []
        
        mode = "DYNAMIC" if use_dynamic else "STATIC"
        print(f"\nEvaluating {mode} fusion...")
        
        for i, query_info in enumerate(self.queries):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(self.queries)}")
            
            query_text = query_info['query_text']
            intent_type = query_info['intent_type']
            ground_truth = query_info['relevant_asin']
            
            try:
                # Retrieve and score products
                products = retrieve_and_score(query_text, self.db_path, self.parser)
                
                if not products:
                    continue
                
                # Apply fusion
                if use_dynamic:
                    constraints = self.parser.parse(query_text)
                    intent, _ = self.classifier.classify(query_text, constraints)
                    ranked = dynamic_fusion(products, intent)
                else:
                    ranked = static_fusion(products)
                
                # Compute relevance
                retrieved_asins = [p['parent_asin'] for p in ranked[:20]]
                relevance = [1.0 if asin == ground_truth else 0.0 for asin in retrieved_asins]
                
                # Metrics
                metrics = {}
                for k in self.k_values:
                    metrics[f'ndcg@{k}'] = ndcg_at_k(relevance, k)
                    metrics[f'precision@{k}'] = precision_at_k(relevance, k)
                metrics['mrr'] = mrr(relevance)
                metrics['intent'] = intent_type
                
                all_results.append(metrics)
                results_by_intent[intent_type].append(metrics)
                
            except Exception as e:
                print(f"Error on query '{query_text}': {e}")
                continue
        
        overall = self._aggregate(all_results)
        by_intent = {intent: self._aggregate(results) 
                     for intent, results in results_by_intent.items()}
        
        return {
            'overall': overall,
            'by_intent': by_intent,
            'num_queries': len(all_results)
        }
    
    def _aggregate(self, results: List[Dict]) -> Dict:
        if not results:
            return {}
        aggregated = {}
        metric_keys = [k for k in results[0].keys() if k != 'intent']
        for key in metric_keys:
            values = [r[key] for r in results]
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'count': len(values)
            }
        return aggregated
    
    def print_results(self, results: Dict, system_name: str):
        print(f"\n{'='*60}")
        print(f"RESULTS: {system_name}")
        print(f"{'='*60}")
        
        overall = results['overall']
        print(f"\nOverall ({results['num_queries']} queries):")
        for metric, stats in overall.items():
            print(f"  {metric:15s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        if 'by_intent' in results:
            print(f"\nBy Intent:")
            for intent, metrics in results['by_intent'].items():
                count = metrics[list(metrics.keys())[0]]['count']
                print(f"\n  {intent.upper()} ({count} queries):")
                for metric, stats in metrics.items():
                    print(f"    {metric:15s}: {stats['mean']:.4f} ± {stats['std']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/reviews-output/product_ranking.sqlite")
    args = parser.parse_args()
    
    evaluator = TextBasedEvaluator(args.db)
    
    # Static fusion baseline
    results_static = evaluator.evaluate(use_dynamic=False)
    evaluator.print_results(results_static, "BASELINE (Static Fusion)")
    
    # Dynamic fusion (intent-aware)
    results_dynamic = evaluator.evaluate(use_dynamic=True)
    evaluator.print_results(results_dynamic, "INTENT-AWARE (Dynamic Fusion)")
    
    # Compute improvement
    print("\n" + "="*60)
    print("IMPROVEMENT: Dynamic over Static")
    print("="*60)
    
    for metric in ['ndcg@5', 'ndcg@10', 'precision@5', 'precision@10', 'mrr']:
        static_val = results_static['overall'][metric]['mean']
        dynamic_val = results_dynamic['overall'][metric]['mean']
        improvement = ((dynamic_val - static_val) / static_val) * 100 if static_val > 0 else 0
        print(f"{metric:15s}: {improvement:+.2f}%")


if __name__ == "__main__":
    main()