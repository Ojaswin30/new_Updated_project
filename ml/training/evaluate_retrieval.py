"""
Evaluation harness for intent-aware multimodal retrieval
Computes NDCG@K and Precision@K using synthetic queries + ground truth
"""
import sqlite3
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import argparse
from pathlib import Path

# Import your pipelines
from src.pipeline.late_fusion import LateFusionPipeline, PipelineConfig as BaseConfig
from ml.training.intent_aware_fusion import IntentAwareFusionPipeline, PipelineConfig as DynamicConfig


# ==================== METRICS ====================

def dcg_at_k(relevance_scores: List[float], k: int) -> float:
    """Discounted Cumulative Gain at K"""
    relevance_scores = relevance_scores[:k]
    if not relevance_scores:
        return 0.0
    
    gains = [(2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevance_scores)]
    return sum(gains)


def ndcg_at_k(relevance_scores: List[float], k: int) -> float:
    """Normalized DCG at K"""
    dcg = dcg_at_k(relevance_scores, k)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal_scores, k)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def precision_at_k(relevance_scores: List[float], k: int, threshold: float = 0.5) -> float:
    """Precision at K (binary relevance)"""
    relevance_scores = relevance_scores[:k]
    if not relevance_scores:
        return 0.0
    
    relevant = sum(1 for score in relevance_scores if score >= threshold)
    return relevant / len(relevance_scores)


def mrr(relevance_scores: List[float], threshold: float = 0.5) -> float:
    """Mean Reciprocal Rank"""
    for i, score in enumerate(relevance_scores, 1):
        if score >= threshold:
            return 1.0 / i
    return 0.0


# ==================== EVALUATION ====================

class RetrievalEvaluator:
    """
    Evaluates retrieval systems using synthetic queries from database
    """
    
    def __init__(self, db_path: str, k_values: List[int] = [5, 10, 20]):
        self.db_path = db_path
        self.k_values = k_values
        
        # Load queries and ground truth
        self.queries = self._load_queries()
        print(f"Loaded {len(self.queries)} evaluation queries")
    
    def _load_queries(self) -> List[Dict]:
        """Load synthetic queries with ground truth from DB"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute("""
            SELECT query_id, query_text, intent_type, relevant_asin
            FROM queries
        """)
        
        queries = [dict(row) for row in c.fetchall()]
        conn.close()
        return queries
    
    def evaluate_system(
        self, 
        pipeline,
        dummy_image: str = "dummy.jpg"
    ) -> Dict:
        """
        Evaluate a retrieval pipeline on all queries
        
        Returns metrics aggregated overall and by intent type
        """
        results_by_intent = defaultdict(list)
        all_results = []
        
        print(f"\nEvaluating on {len(self.queries)} queries...")
        
        for i, query_info in enumerate(self.queries):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{len(self.queries)}")
            
            query_text = query_info['query_text']
            intent_type = query_info['intent_type']
            ground_truth_asin = query_info['relevant_asin']
            
            # Run retrieval (text-only, dummy image)
            try:
                output = pipeline.run(dummy_image, query_text)
                
                retrieved_asins = [r['product_id'] for r in output['results']]
                
                # Compute relevance scores (binary: 1 if ground truth in results, 0 otherwise)
                relevance = [1.0 if asin == ground_truth_asin else 0.0 
                           for asin in retrieved_asins]
                
                # Compute metrics
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
        
        # Aggregate results
        overall = self._aggregate_metrics(all_results)
        by_intent = {
            intent: self._aggregate_metrics(results)
            for intent, results in results_by_intent.items()
        }
        
        return {
            'overall': overall,
            'by_intent': by_intent,
            'num_queries': len(all_results)
        }
    
    def _aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Compute mean and std of metrics"""
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
        """Pretty print evaluation results"""
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
                print(f"\n  {intent.upper()} ({metrics[list(metrics.keys())[0]]['count']} queries):")
                for metric, stats in metrics.items():
                    print(f"    {metric:15s}: {stats['mean']:.4f} ± {stats['std']:.4f}")


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser("Evaluate retrieval systems")
    parser.add_argument(
        "--db",
        default="data/reviews-output/product_ranking.sqlite",
        help="Path to evaluation database"
    )
    parser.add_argument(
        "--systems",
        nargs='+',
        default=['baseline', 'dynamic'],
        choices=['baseline', 'dynamic'],
        help="Which systems to evaluate"
    )
    parser.add_argument(
        "--dummy-image",
        default="dummy.jpg",
        help="Dummy image path for text-only evaluation"
    )
    args = parser.parse_args()
    
    evaluator = RetrievalEvaluator(args.db)
    
    # Evaluate baseline (static fusion)
    if 'baseline' in args.systems:
        print("\n" + "="*60)
        print("EVALUATING BASELINE (Static Fusion)")
        print("="*60)
        config_baseline = BaseConfig(catalog_db_path=args.db)
        pipeline_baseline = LateFusionPipeline(config_baseline)
        results_baseline = evaluator.evaluate_system(pipeline_baseline, args.dummy_image)
        evaluator.print_results(results_baseline, "BASELINE (Static Fusion)")
    
    # Evaluate dynamic fusion
    if 'dynamic' in args.systems:
        print("\n" + "="*60)
        print("EVALUATING DYNAMIC FUSION (Intent-Aware)")
        print("="*60)
        config_dynamic = DynamicConfig(
            catalog_db_path=args.db,
            use_dynamic_fusion=True
        )
        pipeline_dynamic = IntentAwareFusionPipeline(config_dynamic)
        results_dynamic = evaluator.evaluate_system(pipeline_dynamic, args.dummy_image)
        evaluator.print_results(results_dynamic, "INTENT-AWARE (Dynamic Fusion)")


if __name__ == "__main__":
    main()