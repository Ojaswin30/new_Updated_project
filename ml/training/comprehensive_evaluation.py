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
from training.intent_classifier import IntentClassifier


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
    variation = hash_val / 1000.0

    return max(0.0, min(1.0, base_score + variation))


def retrieve_and_score(query_text: str, db_path: str, parser: ConstraintParser) -> List[Dict]:
    constraints = parser.parse(query_text)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # SQL pre-filter: keyword match on title to narrow candidates
    # Avoids scanning all 47K products for every query
    words = [w for w in query_text.lower().split() if len(w) > 2][:4]

    if words:
        where = " OR ".join(["LOWER(p.title) LIKE ?" for w in words])
        params = [f"%{w}%" for w in words]
        sql = f"""
            SELECT p.parent_asin, p.title, p.category,
                   COALESCE(r.review_score, 0.0) AS review_score
            FROM products p
            LEFT JOIN product_ranking r ON p.parent_asin = r.parent_asin
            WHERE {where}
            LIMIT 500
        """
        c.execute(sql, params)
    else:
        c.execute("""
            SELECT p.parent_asin, p.title, p.category,
                   COALESCE(r.review_score, 0.0) AS review_score
            FROM products p
            LEFT JOIN product_ranking r ON p.parent_asin = r.parent_asin
            LIMIT 200
        """)

    all_products = [dict(row) for row in c.fetchall()]
    conn.close()

    scored = []
    for product in all_products:
        text_score = word_overlap_score(query_text, product['title'])
        visual_score = compute_visual_score(constraints, product['title'], product['parent_asin'])
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


def method3_intent_aware_dynamic_fusion(
    products: List[Dict],
    query_text: str,
    constraints: Dict,
    classifier: IntentClassifier
) -> List[Dict]:
    """
    Method 3: Intent-Aware Dynamic Fusion
    Adaptive weights based on query intent
    """
    intent, scores = classifier.classify(query_text, constraints)

    visual_conf    = scores.get('visual', 0.5)
    attribute_conf = scores.get('attribute', 0.5)

    if intent == 'visual':
        alpha, beta, gamma = 0.70, 0.15, 0.15
    elif intent == 'attribute':
        alpha, beta, gamma = 0.15, 0.65, 0.20
    else:  # hybrid
        alpha = 0.35 + 0.20 * visual_conf
        beta  = 0.35 + 0.20 * attribute_conf
        gamma = 0.15
        total = alpha + beta + gamma
        alpha, beta, gamma = alpha / total, beta / total, gamma / total

    ranked = []
    for p in products:
        final_score = (
            alpha * p['visual_score'] +
            beta  * p['text_score'] +
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
    def __init__(self, db_path: str, k_values: List[int] = [5, 10, 20], limit: int = None):
        self.db_path = db_path
        self.k_values = k_values
        self.limit = limit
        self.parser = ConstraintParser()
        self.classifier = IntentClassifier()   # single shared instance

        self.queries = self._load_queries()
        print(f"Loaded {len(self.queries)} evaluation queries\n")

    def _load_queries(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        if self.limit:
            # Stratified sample: equal representation across all 3 intent types
            queries = []
            per_intent = self.limit // 3
            for intent in ['visual', 'hybrid', 'attribute']:
                c.execute(
                    "SELECT * FROM queries WHERE intent_type = ? ORDER BY RANDOM() LIMIT ?",
                    (intent, per_intent)
                )
                queries.extend([dict(row) for row in c.fetchall()])
        else:
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

            query_text  = query_info['query_text']
            intent_type = query_info['intent_type']
            ground_truth = query_info['relevant_asin']

            try:
                products = retrieve_and_score(query_text, self.db_path, self.parser)
                if not products:
                    continue

                constraints = self.parser.parse(query_text)

                if method_name == 'symbolic_early':
                    ranked = method1_symbolic_early_fusion(products)
                elif method_name == 'static_late':
                    ranked = method2_static_late_fusion(products)
                else:  # intent_aware_dynamic
                    ranked = method3_intent_aware_dynamic_fusion(
                        products, query_text, constraints, self.classifier
                    )

                retrieved_asins = [p['parent_asin'] for p in ranked[:20]]
                relevance = [1.0 if asin == ground_truth else 0.0
                             for asin in retrieved_asins]

                metrics = {}
                for k in self.k_values:
                    metrics[f'ndcg@{k}']      = ndcg_at_k(relevance, k)
                    metrics[f'precision@{k}'] = precision_at_k(relevance, k)
                metrics['mrr']    = mrr(relevance)
                metrics['intent'] = intent_type

                all_results.append(metrics)
                results_by_intent[intent_type].append(metrics)

            except Exception as e:
                continue

        overall   = self._aggregate(all_results)
        by_intent = {intent: self._aggregate(results)
                     for intent, results in results_by_intent.items()}

        return {
            'overall':     overall,
            'by_intent':   by_intent,
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
                'std':  np.std(values)
            }
        return aggregated
    def bootstrap_significance_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        n_iterations: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Paired bootstrap test to check if the difference between two methods is significant.
        Returns p-value, observed difference, and whether result is significant.
        """
        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)
        observed_diff = np.mean(scores_b) - np.mean(scores_a)

        n = len(scores_a)
        bootstrap_diffs = []

        rng = np.random.default_rng(seed=42)
        for _ in range(n_iterations):
            indices = rng.integers(0, n, size=n)
            boot_a = scores_a[indices]
            boot_b = scores_b[indices]
            bootstrap_diffs.append(np.mean(boot_b) - np.mean(boot_a))

        bootstrap_diffs = np.array(bootstrap_diffs)

        # Two-tailed p-value: proportion of bootstrap diffs on the opposite side of zero
        if observed_diff >= 0:
            p_value = np.mean(bootstrap_diffs <= 0) * 2
        else:
            p_value = np.mean(bootstrap_diffs >= 0) * 2
        p_value = min(p_value, 1.0)

        alpha = 1.0 - confidence_level
        ci_low  = np.percentile(bootstrap_diffs, (alpha / 2) * 100)
        ci_high = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)

        return {
            'observed_diff': observed_diff,
            'p_value':       p_value,
            'ci_low':        ci_low,
            'ci_high':       ci_high,
            'significant':   p_value < alpha,
            'n_iterations':  n_iterations,
        }

    def collect_per_query_scores(self, method_name: str, metric: str = 'ndcg@10') -> List[float]:
        """
        Re-runs evaluation for one method and returns a per-query list of metric scores.
        Required for paired bootstrap — each element corresponds to the same query.
        """
        scores = []
        for query_info in self.queries:
            query_text   = query_info['query_text']
            ground_truth = query_info['relevant_asin']

            try:
                products = retrieve_and_score(query_text, self.db_path, self.parser)
                if not products:
                    scores.append(0.0)
                    continue

                constraints = self.parser.parse(query_text)

                if method_name == 'symbolic_early':
                    ranked = method1_symbolic_early_fusion(products)
                elif method_name == 'static_late':
                    ranked = method2_static_late_fusion(products)
                else:
                    ranked = method3_intent_aware_dynamic_fusion(
                        products, query_text, constraints, self.classifier
                    )

                retrieved_asins = [p['parent_asin'] for p in ranked[:20]]
                relevance = [1.0 if asin == ground_truth else 0.0
                            for asin in retrieved_asins]

                k = int(metric.split('@')[1]) if '@' in metric else 10
                scores.append(ndcg_at_k(relevance, k))

            except Exception:
                scores.append(0.0)

        return scores

    def print_bootstrap_results(self, results_dict: Dict[str, Dict]):
        """
        Runs paired bootstrap tests for all method pairs and prints a significance table.
        """
        print("\n" + "=" * 80)
        print("🔬 BOOTSTRAP SIGNIFICANCE TESTS  (n=1000 iterations, 95% confidence)")
        print("=" * 80)
        print("Metric: NDCG@10  |  H0: no difference between methods")
        print("-" * 80)

        method_pairs = [
            ('symbolic_early',  'static_late',         'Static Late   vs Symbolic Early'),
            ('symbolic_early',  'intent_aware_dynamic', 'Intent-Aware  vs Symbolic Early'),
            ('static_late',     'intent_aware_dynamic', 'Intent-Aware  vs Static Late   '),
        ]

        print("Collecting per-query scores for bootstrap (this may take a moment)...")
        score_cache: Dict[str, List[float]] = {}
        for name in ['symbolic_early', 'static_late', 'intent_aware_dynamic']:
            score_cache[name] = self.collect_per_query_scores(name, metric='ndcg@10')

        print(f"\n{'Comparison':<42} {'Δ NDCG@10':>10} {'p-value':>10} {'95% CI':>22} {'Sig?':>6}")
        print("-" * 95)

        for method_a, method_b, label in method_pairs:
            result = self.bootstrap_significance_test(
                score_cache[method_a],
                score_cache[method_b],
                n_iterations=1000,
                confidence_level=0.95,
            )
            sig_marker = "✅ YES" if result['significant'] else "❌ NO "
            ci_str = f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]"
            print(
                f"{label:<42} "
                f"{result['observed_diff']:>+10.4f} "
                f"{result['p_value']:>10.4f} "
                f"{ci_str:>22} "
                f"{sig_marker:>6}"
            )

        print("-" * 95)
        print("  Δ > 0 means the second method is better than the first.")
        print("  Significant = p < 0.05 (difference unlikely due to chance).\n")
    def print_comparison_table(self, results_dict: Dict[str, Dict]):
        print("\n" + "=" * 80)
        print("COMPREHENSIVE COMPARISON: 3 FUSION METHODS")
        print("=" * 80)

        # Overall comparison
        print("\n📊 OVERALL PERFORMANCE")
        print("-" * 80)
        print(f"{'Metric':<15} {'Symbolic Early':<20} {'Static Late':<20} {'Intent-Aware':<20}")
        print("-" * 80)

        metrics = ['ndcg@5', 'ndcg@10', 'ndcg@20', 'precision@5', 'precision@10', 'mrr']

        for metric in metrics:
            symbolic = results_dict['symbolic_early']['overall'][metric]['mean']
            static   = results_dict['static_late']['overall'][metric]['mean']
            dynamic  = results_dict['intent_aware_dynamic']['overall'][metric]['mean']
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
        print("\n" + "=" * 80)
        print("📈 IMPROVEMENT ANALYSIS (vs Symbolic Early Fusion)")
        print("=" * 80)

        baseline = results_dict['symbolic_early']['overall']

        for method_name in ['static_late', 'intent_aware_dynamic']:
            print(f"\n{method_name.replace('_', ' ').title()}:")
            for metric in ['ndcg@10', 'precision@10', 'mrr']:
                base_val = baseline[metric]['mean']
                new_val  = results_dict[method_name]['overall'][metric]['mean']
                improvement = ((new_val - base_val) / base_val) * 100 if base_val > 0 else 0
                sign = "✅" if improvement > 0 else "❌" if improvement < 0 else "="
                print(f"  {sign} {metric:<12}: {improvement:+6.2f}%")
        
        self.print_bootstrap_results(results_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/reviews-output/product_ranking_v3.sqlite")
    parser.add_argument("--limit", type=int, default=1500,
                        help="Queries to evaluate (stratified across intent types). Default 1500 (~500 per intent).")
    args = parser.parse_args()

    evaluator = ComprehensiveEvaluator(args.db, limit=args.limit)

    results = {}

    results['symbolic_early']       = evaluator.evaluate_method('symbolic_early')
    print()
    results['static_late']          = evaluator.evaluate_method('static_late')
    print()
    results['intent_aware_dynamic'] = evaluator.evaluate_method('intent_aware_dynamic')
    print()

    evaluator.print_comparison_table(results)


if __name__ == "__main__":
    main()