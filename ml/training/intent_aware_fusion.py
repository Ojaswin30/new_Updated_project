"""
Intent-Aware Dynamic Fusion Pipeline
Extends late_fusion.py with adaptive weight adjustment based on query intent
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from src.pipeline.constraint_parser import ConstraintParser
from src.pipeline.query_builder import SimpleQueryBuilder as SQLQueryBuilder, QueryBuildResult
from src.vision.early_fusion_clip_inference import early_fusion_image_infer
from training.intent_classifier import IntentClassifier


# ----------------------------
# Data Contracts
# ----------------------------

@dataclass
class PipelineConfig:
    catalog_db_path: str = "data/reviews-output/product_ranking.sqlite"
    products_table: str = "products"
    ranking_table: str = "product_ranking"

    sql_limit: int = 200
    top_k_return: int = 20

    # Static weights (baseline)
    alpha_visual: float = 0.50
    beta_constraints: float = 0.35
    gamma_review: float = 0.15
    
    # Dynamic fusion enabled?
    use_dynamic_fusion: bool = True


# ----------------------------
# Intent-Aware Dynamic Fusion Pipeline
# ----------------------------

class IntentAwareFusionPipeline:
    """
    Novel contribution: Adaptive fusion weight adjustment based on query intent
    
    Instead of fixed α, β, γ weights, we dynamically adjust based on:
    - Visual-dominant queries → increase α (visual weight)
    - Attribute-dominant queries → increase β (constraint weight)
    - Hybrid queries → balanced weights
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        self.parser = ConstraintParser()
        self.intent_classifier = IntentClassifier()
        self.sql_builder = SQLQueryBuilder(
            table_products=self.config.products_table,
            table_ranking=self.config.ranking_table,
            join_ranking=True,
        )

    # ----------------------------
    # Public API
    # ----------------------------

    def run(self, image_path: str, text: str) -> Dict[str, Any]:
        """Main pipeline execution"""
        vision_out = self._run_vision(image_path)
        text_constraints = self.parser.parse(text)

        # NEW: Intent classification
        intent, intent_scores = self.intent_classifier.classify(text, text_constraints)

        merged_constraints = self._merge_constraints(
            vision_out, text_constraints
        )

        query_result = self.sql_builder.build(
            constraints=merged_constraints,
            limit=self.config.sql_limit,
            sort_by="review_desc",
        )

        candidates = self._execute_sql(query_result)
        
        # NEW: Dynamic weight computation
        fusion_weights = self._compute_fusion_weights(intent, intent_scores)
        
        ranked = self._rank_candidates(
            candidates, merged_constraints, fusion_weights
        )

        return {
            "constraints_text": text_constraints,
            "constraints_vision": vision_out,
            "constraints_merged": merged_constraints,
            "intent": intent,
            "intent_scores": intent_scores,
            "fusion_weights": fusion_weights,
            "sql": query_result.sql,
            "sql_params": query_result.params,
            "sql_debug": query_result.debug,
            "results": ranked[: self.config.top_k_return],
        }

    # ----------------------------
    # Vision (CLIP)
    # ----------------------------

    def _run_vision(self, image_path: str) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            return {"category": None, "color": None, "visual_score": 0.0}

        image_pred = early_fusion_image_infer(image_path)

        return {
            "category": image_pred.get("category"),
            "color": image_pred.get("color"),
            "visual_score": float(image_pred.get("category_score", 0.0)),
        }

    # ----------------------------
    # Constraint Merge
    # ----------------------------

    def _merge_constraints(self, vision: Dict, text: Dict) -> Dict:
        merged = dict(text)

        if not merged.get("category") and vision.get("category"):
            merged["category"] = vision["category"]

        if not merged.get("color") and vision.get("color"):
            merged["color"] = vision["color"]

        merged["_vision_score"] = vision.get("visual_score", 0.0)
        return merged

    # ----------------------------
    # SQL Execution
    # ----------------------------

    def _execute_sql(self, query: QueryBuildResult) -> List[Dict[str, Any]]:
        if not os.path.exists(self.config.catalog_db_path):
            raise FileNotFoundError(
                f"DB not found: {self.config.catalog_db_path}"
            )

        conn = sqlite3.connect(self.config.catalog_db_path)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            cur.execute(query.sql, query.params)
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    # ----------------------------
    # NOVEL: Dynamic Fusion Weight Computation
    # ----------------------------

    def _compute_fusion_weights(self, intent: str, intent_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Core novelty: Compute adaptive fusion weights based on detected intent
        
        Strategy:
        - visual-dominant → boost α (visual weight)
        - attribute-dominant → boost β (constraint weight)
        - hybrid → balanced weights
        - γ (review) remains relatively stable as a tiebreaker
        """
        if not self.config.use_dynamic_fusion:
            # Baseline: static weights
            return {
                'alpha': self.config.alpha_visual,
                'beta': self.config.beta_constraints,
                'gamma': self.config.gamma_review
            }
        
        # Dynamic weight adjustment
        base_alpha = self.config.alpha_visual
        base_beta = self.config.beta_constraints
        base_gamma = self.config.gamma_review
        
        if intent == 'visual':
            # Boost visual, reduce constraint
            alpha = min(0.70, base_alpha + 0.20)
            beta = max(0.15, base_beta - 0.15)
            gamma = base_gamma
        
        elif intent == 'attribute':
            # Boost constraint, reduce visual
            alpha = max(0.20, base_alpha - 0.20)
            beta = min(0.60, base_beta + 0.20)
            gamma = base_gamma
        
        else:  # hybrid
            # Moderate adjustment toward balance
            visual_conf = intent_scores.get('visual', 0.5)
            attribute_conf = intent_scores.get('attribute', 0.5)
            
            # Interpolate based on confidence
            alpha = base_alpha + 0.10 * (visual_conf - 0.5)
            beta = base_beta + 0.10 * (attribute_conf - 0.5)
            gamma = base_gamma
        
        # Normalize to sum to 1.0
        total = alpha + beta + gamma
        alpha /= total
        beta /= total
        gamma /= total
        
        return {'alpha': alpha, 'beta': beta, 'gamma': gamma}

    # ----------------------------
    # Ranking with Dynamic Weights
    # ----------------------------

    def _rank_candidates(
        self, 
        candidates: List[Dict], 
        constraints: Dict,
        fusion_weights: Dict[str, float]
    ) -> List[Dict]:
        """Rank candidates using dynamic fusion weights"""
        ranked = []

        alpha = fusion_weights['alpha']
        beta = fusion_weights['beta']
        gamma = fusion_weights['gamma']

        for row in candidates:
            visual = float(constraints.get("_vision_score", 0.0))
            constraint = self._constraint_match_score(row, constraints)
            review = float(row.get("review_score", 0.0))

            final = (
                alpha * visual
                + beta * constraint
                + gamma * review
            )

            ranked.append({
                "product_id": row.get("parent_asin"),
                "title": row.get("title"),
                "final_score": final,
                "visual_score": visual,
                "constraint_score": constraint,
                "review_score": review,
                "fields": row,
            })

        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        return ranked

    def _constraint_match_score(self, row: Dict, constraints: Dict) -> float:
        score, max_score = 0.0, 0.0

        def match_fuzzy(field: str, w: float):
            nonlocal score, max_score
            val = constraints.get(field)
            if val is None:
                return
            max_score += w
            row_val = str(row.get(field, "") or "").lower()
            # Use partial match instead of exact equality
            if str(val).lower() in row_val or row_val in str(val).lower():
                score += w
            elif row_val:
                score += w * 0.3   # partial credit for non-null field

        match_fuzzy("category", 1.0)
        match_fuzzy("color", 1.0)
        match_fuzzy("size", 0.8)
        match_fuzzy("material", 0.8)

        keywords = constraints.get("keywords") or []
        if keywords:
            max_score += 1.0
            title = str(row.get("title", "")).lower()
            hits = sum(1 for k in keywords if k.lower() in title)
            score += min(1.0, hits / len(keywords))

        price_max = constraints.get("price_max")
        if price_max:
            max_score += 0.5
            price = row.get("price")
            if price is None:
                score += 0.25   # neutral for unknown price
            elif float(price) <= float(price_max):
                score += 0.5

        return 0.0 if max_score == 0 else min(1.0, score / max_score)


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser("Intent-aware dynamic fusion pipeline")
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument(
        "--db",
        default="data/reviews-output/product_ranking.sqlite"
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Use static weights (baseline mode)"
    )
    args = parser.parse_args()

    config = PipelineConfig(
        catalog_db_path=args.db,
        use_dynamic_fusion=not args.static
    )
    
    pipe = IntentAwareFusionPipeline(config)

    out = pipe.run(args.image, args.text)

    print("\n--- INTENT-AWARE FUSION OUTPUT ---")
    print(f"Query Intent: {out['intent']}")
    print(f"Intent Scores: {out['intent_scores']}")
    print(f"Fusion Weights: α={out['fusion_weights']['alpha']:.3f}, "
          f"β={out['fusion_weights']['beta']:.3f}, "
          f"γ={out['fusion_weights']['gamma']:.3f}")
    print("\nTop results:")
    for i, r in enumerate(out["results"], 1):
        print(
            f"{i:02d}. {r['product_id']} | score={r['final_score']:.4f} "
            f"(visual={r['visual_score']:.3f}, "
            f"constraint={r['constraint_score']:.3f}, "
            f"review={r['review_score']:.3f}) | {r['title']}"
        )


if __name__ == "__main__":
    main()