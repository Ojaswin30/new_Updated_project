"""
COMPREHENSIVE EVALUATION: Compare 3 Fusion Methods
1. Symbolic Early Fusion (baseline)
2. Static Late Fusion (fixed weights)
3. Intent-Aware Dynamic Fusion (adaptive weights)
"""
import sqlite3
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import argparse
import re

from src.pipeline.constraint_parser import ConstraintParser


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
    return dcg / idcg if idcg > 0 else 0.0

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


# ==================== DYNAMIC INTENT CLASSIFIER ====================

class DynamicIntentClassifier:
    """Learns weights from query content dynamically"""
    
    def __init__(self):
        self.colors = {
            'descriptive': ['bright red', 'dark blue', 'light green', 'deep purple'],
            'nuanced': ['navy', 'burgundy', 'teal', 'beige', 'maroon', 'olive'],
            'basic': ['red', 'blue', 'green', 'black', 'white', 'yellow', 'pink', 'purple']
        }
        self.visual_descriptors = ['striped', 'floral', 'polka dot', 'vintage', 'elegant', 'modern']
        self.materials = ['cotton', 'polyester', 'silk', 'wool', 'leather', 'denim']
        self.technical_specs = ['waterproof', 'breathable', 'stretch', 'wrinkle-free']
    
    def _compute_visual_strength(self, query_text: str, constraints: Dict) -> float:
        query_lower = query_text.lower()
        strength = 0.0
        
        color = constraints.get('color')
        if color:
            if any(color in desc for desc in self.colors['descriptive']):
                strength += 0.35
            elif color in self.colors['nuanced']:
                strength += 0.25
            elif color in self.colors['basic']:
                strength += 0.20
        
        descriptor_count = sum(1 for desc in self.visual_descriptors if desc in query_lower)
        strength += min(0.30, descriptor_count * 0.10)
        
        if constraints.get('category'):
            strength += 0.15
        
        return min(1.0, strength)
    
    def _compute_attribute_strength(self, query_text: str, constraints: Dict) -> float:
        query_lower = query_text.lower()
        strength = 0.0
        
        if constraints.get('size'):
            strength += 0.25
        
        if constraints.get('material'):
            strength += 0.20
        
        price_patterns = ['under', 'below', 'less than', 'budget', 'price']
        if any(p in query_lower for p in price_patterns):
            strength += 0.20
        
        spec_count = sum(1 for spec in self.technical_specs if spec in query_lower)
        strength += min(0.25, spec_count * 0.15)
        
        numeric_count = len(re.findall(r'\b\d+\b', query_text))
        strength += min(0.15, numeric_count * 0.08)
        
        return min(1.0, strength)
    
    def classify(self, query_text: str, constraints: Dict) -> Tuple[str, Dict]:
        visual_strength = self._compute_visual_strength(query_text, constraints)
        attribute_strength = self._compute_attribute_strength(query_text, constraints)
        
        total = visual_strength + attribute_strength
        if total == 0:
            return 'hybrid', {'visual_score': 0.5, 'attribute_score': 0.5, 'confidence': 0.3}
        
        visual_ratio = visual_strength / total
        attribute_ratio = attribute_strength / total
        
        separation = abs(visual_ratio - attribute_ratio)
        query_length = len(query_text.split())
        confidence = separation * (0.5 + 0.5 * min(query_length / 5.0, 1.0))
        
        threshold = 0.65 if confidence > 0.6 else 0.70
        
        if visual_ratio >= threshold:
            intent = 'visual'
        elif attribute_ratio >= threshold:
            intent = 'attribute'
        else:
            intent = 'hybrid'
        
        return intent, {
            'visual_score': visual_ratio,
            'attribute_score': attribute_ratio,
            'confidence': confidence
        }
    
    def get_fusion_weights(self, intent: str, scores: Dict) -> Tuple[float, float, float]:
        visual_score = scores['visual_score']
        attribute_score = scores['attribute_score']
        confidence = scores['confidence']
        
        if intent == 'visual':
            alpha = 0.70 + (0.15 * confidence)
            beta = 0.20 - (0.10 * confidence)
            gamma = 0.10
        elif intent == 'attribute':
            beta = 0.65 + (0.15 * confidence)
            alpha = 0.20 - (0.10 * confidence)
            gamma = 0.15
        else:
            alpha = 0.30 + (0.30 * visual_score)
            beta = 0.30 + (0.30 * attribute_score)
            gamma = 0.20
        
        total = alpha + beta + gamma
        return alpha/total, beta/total, gamma/total


# ==================== SCORING ====================

def word_overlap_score(query: str, title: str) -> float:
    query_words = set(query.lower().split())
    title_words = set(title.lower().split())
    if not query_words:
        return 0.0
    return len(query_words & title_words) / len(query_words)


def compute_visual_score(constraints: Dict, title: str, product_id: str) -> float:
    title_lower = title.lower()
    base_score = 0.50
    
    query_color = constraints.get('color')
    if query_color:
        base_score = 0.90 if query_color.lower() in title_lower else 0.10
    
    query_category = constraints.get('category')
    if query_category:
        if query_category.lower() in title_lower:
            base_score = max(base_score, 0.70)
        else:
            base_score = min(base_score, 0.30)
    
    # Tie-breaking variation
    hash_val = hash(product_id) % 100
    variation = (hash_val / 1000.0)
    
    return max(0.0, min(1.0, base_score + variation))


def retrieve_and_score(query_text: str, db_path: str, parser: ConstraintParser) -> List[Dict]:
    constraints = parser.parse(query_text)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("""
        SELECT p.parent_asin, p.title, p.category,
               COALESCE(r.review_score, 0.0) AS review_score
        FROM products p
        LEFT JOIN product_ranking r ON p.parent_asin = r.parent_asin
    """)
    
    all_products = [dict(row) for row in c.fetchall()]
    conn.close()
    
    scored = []
    for product in all_products:
        text_score = word_overlap_score(query_text, product['title'])
        visual_score = compute_visual_score(constraints, product['title'], product['parent_asin'])
        
        if text_score > 0 or visual_score > 0.6:
            product['text_score'] = text_score
            product['visual_score'] = visual_score
            scored.append(product)
    
    return scored


# ==================== 3 FUSION METHODS ====================

def method1_symbolic_early_fusion(products: List[Dict]) -> List[Dict]:
    """
    Method 1: Symbolic Early Fusion
    Rule-based scoring with equal weights
    """
    alpha, beta, gamma = 0.33, 0.33, 0.34
    
    ranked = []
    for p in products:
        final_score = (
            alpha * p['visual_score'] +
            beta * p['text_score'] +
            gamma * p['review_score']
        )
        p['final_score'] = final_score
        p['method'] = 'symbolic_early'
        ranked.append(p)
    
    ranked.sort(key=lambda x: x['final_score'], reverse=True)
    return ranked[:200]


def method2_static_late_fusion(products: List[Dict]) -> List[Dict]:
    """
    Method 2: Static Late Fusion
    Fixed weights optimized for general case
    """
    alpha, beta, gamma = 0.40, 0.40, 0.20
    
    ranked = []
    for p in products:
        final_score = (
            alpha * p['visual_score'] +
            beta * p['text_score'] +
            gamma * p['review_score']
        )
        p['final_score'] = final_score
        p['method'] = 'static_late'
        ranked.append(p)
    
    ranked.sort(key=lambda x: x['final_score'], reverse=True)
    return ranked[:200]


def method3_intent_aware_dynamic_fusion(products: List[Dict], query_text: str, 
                                       constraints: Dict, classifier: DynamicIntentClassifier) -> List[Dict]:
    """
    Method 3: Intent-Aware Dynamic Fusion
    Adaptive weights based on query intent
    """
    intent, scores = classifier.classify(query_text, constraints)
    alpha, beta, gamma = classifier.get_fusion_weights(intent, scores)
    
    ranked = []
    for p in products:
        final_score = (
            alpha * p['visual_score'] +
            beta * p['text_score'] +
            gamma * p['review_score']
        )
        p['final_score'] = final_score
        p['method'] = f'dynamic_{intent}'
        p['weights'] = f'α={alpha:.2f},β={beta:.2f},γ={gamma:.2f}'
        ranked.append(p)
    
    ranked.sort(key=lambda x: x['final_score'], reverse=True)
    return ranked[:200]


# ==================== EVALUATION ====================

class ComprehensiveEvaluator:
    def __init__(self, db_path: str, k_values: List[int] = [5, 10, 20]):
        self.db_path = db_path
        self.k_values = k_values
        self.parser = ConstraintParser()
        self.classifier = DynamicIntentClassifier()
        
        self.queries = self._load_queries()
        print(f"Loaded {len(self.queries)} evaluation queries\n")
    
    def _load_queries(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM queries")
        queries = [dict(row) for row in c.fetchall()]
        conn.close()
        return queries
    
    def evaluate_method(self, method_name: str) -> Dict:
        results_by_intent = defaultdict(list)
        all_results = []
        
        print(f"Evaluating {method_name}...")
        
        for i, query_info in enumerate(self.queries):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(self.queries)}")
            
            query_text = query_info['query_text']
            intent_type = query_info['intent_type']
            ground_truth = query_info['relevant_asin']
            
            try:
                products = retrieve_and_score(query_text, self.db_path, self.parser)
                if not products:
                    continue
                
                constraints = self.parser.parse(query_text)
                
                # Apply appropriate fusion method
                if method_name == 'symbolic_early':
                    ranked = method1_symbolic_early_fusion(products)
                elif method_name == 'static_late':
                    ranked = method2_static_late_fusion(products)
                else:  # intent_aware_dynamic
                    ranked = method3_intent_aware_dynamic_fusion(
                        products, query_text, constraints, self.classifier
                    )
                
                retrieved_asins = [p['parent_asin'] for p in ranked[:20]]
                relevance = [1.0 if asin == ground_truth else 0.0 for asin in retrieved_asins]
                
                metrics = {}
                for k in self.k_values:
                    metrics[f'ndcg@{k}'] = ndcg_at_k(relevance, k)
                    metrics[f'precision@{k}'] = precision_at_k(relevance, k)
                metrics['mrr'] = mrr(relevance)
                metrics['intent'] = intent_type
                
                all_results.append(metrics)
                results_by_intent[intent_type].append(metrics)
                
            except Exception as e:
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
                'std': np.std(values)
            }
        return aggregated
    
    def print_comparison_table(self, results_dict: Dict[str, Dict]):
        print("\n" + "="*80)
        print("COMPREHENSIVE COMPARISON: 3 FUSION METHODS")
        print("="*80)
        
        # Overall comparison
        print("\n📊 OVERALL PERFORMANCE")
        print("-" * 80)
        print(f"{'Metric':<15} {'Symbolic Early':<20} {'Static Late':<20} {'Intent-Aware':<20}")
        print("-" * 80)
        
        metrics = ['ndcg@5', 'ndcg@10', 'ndcg@20', 'precision@5', 'precision@10', 'mrr']
        
        for metric in metrics:
            symbolic = results_dict['symbolic_early']['overall'][metric]['mean']
            static = results_dict['static_late']['overall'][metric]['mean']
            dynamic = results_dict['intent_aware_dynamic']['overall'][metric]['mean']
            
            print(f"{metric:<15} {symbolic:>6.4f}             {static:>6.4f}             {dynamic:>6.4f}")
        
        # By intent comparison
        for intent in ['visual', 'attribute', 'hybrid']:
            print(f"\n📊 {intent.upper()} QUERIES")
            print("-" * 80)
            
            for metric in ['ndcg@10', 'precision@10']:
                values = []
                for method in ['symbolic_early', 'static_late', 'intent_aware_dynamic']:
                    if intent in results_dict[method]['by_intent']:
                        val = results_dict[method]['by_intent'][intent][metric]['mean']
                        values.append(f"{val:>6.4f}")
                    else:
                        values.append("  N/A ")
                
                print(f"{metric:<15} {values[0]:<20} {values[1]:<20} {values[2]:<20}")
        
        # Improvement analysis
        print("\n" + "="*80)
        print("📈 IMPROVEMENT ANALYSIS (vs Symbolic Early Fusion)")
        print("="*80)
        
        baseline = results_dict['symbolic_early']['overall']
        
        for method_name in ['static_late', 'intent_aware_dynamic']:
            print(f"\n{method_name.replace('_', ' ').title()}:")
            for metric in ['ndcg@10', 'precision@10', 'mrr']:
                base_val = baseline[metric]['mean']
                new_val = results_dict[method_name]['overall'][metric]['mean']
                improvement = ((new_val - base_val) / base_val) * 100 if base_val > 0 else 0
                sign = "✅" if improvement > 0 else "❌" if improvement < 0 else "="
                print(f"  {sign} {metric:<12}: {improvement:+6.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/reviews-output/product_ranking.sqlite")
    args = parser.parse_args()
    
    evaluator = ComprehensiveEvaluator(args.db)
    
    # Evaluate all 3 methods
    results = {}
    
    results['symbolic_early'] = evaluator.evaluate_method('symbolic_early')
    print()
    
    results['static_late'] = evaluator.evaluate_method('static_late')
    print()
    
    results['intent_aware_dynamic'] = evaluator.evaluate_method('intent_aware_dynamic')
    print()
    
    # Print comprehensive comparison
    evaluator.print_comparison_table(results)


if __name__ == "__main__":
    main()