from __future__ import annotations

import argparse
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from ml.src.pipeline.constraint_parser import ConstraintParser
from ml.src.pipeline.query_builder import SimpleQueryBuilder as SQLQueryBuilder, QueryBuildResult
from ml.src.vision.early_fusion_clip_inference import early_fusion_image_infer


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

    alpha_visual: float = 0.50
    beta_constraints: float = 0.35
    gamma_review: float = 0.15


# ----------------------------
# Late Fusion Pipeline
# ----------------------------

class LateFusionPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        self.parser = ConstraintParser()
        self.sql_builder = SQLQueryBuilder(
            table_products=self.config.products_table,
            table_ranking=self.config.ranking_table,
            join_ranking=True,
        )

    # ----------------------------
    # Public API
    # ----------------------------

    def run(self, image_path: str, text: str) -> Dict[str, Any]:
        vision_out = self._run_vision(image_path)
        text_constraints = self.parser.parse(text)

        merged_constraints = self._merge_constraints(
            vision_out, text_constraints
        )

        query_result = self.sql_builder.build(
            constraints=merged_constraints,
            limit=self.config.sql_limit,
            sort_by="review_desc",
        )

        # candidates = self._execute_sql(query_result)
        candidates = []
        ranked = self._rank_candidates(
            candidates, merged_constraints
        )

        return {
            "constraints_text": text_constraints,
            "constraints_vision": vision_out,
            "constraints_merged": merged_constraints,
            "sql": query_result.sql,
            "sql_params": query_result.params,
            "sql_debug": query_result.debug,
            "results": ranked[: self.config.top_k_return],
        }

    # ----------------------------
    # Vision (FAST CLIP)
    # ----------------------------

    def _run_vision(self, image_path: str) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_pred = early_fusion_image_infer(image_path)

        return {
            "category": image_pred.get("category"),
            "color": image_pred.get("color"),
            "visual_score": float(image_pred.get("category_score", 0.0)),
        }

    # ----------------------------
    # Constraint Merge (Late Fusion)
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
    # Late Fusion Ranking
    # ----------------------------

    def _rank_candidates(self, candidates: List[Dict], constraints: Dict) -> List[Dict]:
        ranked = []

        for row in candidates:
            visual = float(constraints.get("_vision_score", 0.0))
            constraint = self._constraint_match_score(row, constraints)
            review = float(row.get("review_score", 0.0))

            final = (
                self.config.alpha_visual * visual
                + self.config.beta_constraints * constraint
                + self.config.gamma_review * review
            )

            ranked.append({
                "product_id": row.get("product_id"),
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

        def match(field: str, w: float):
            nonlocal score, max_score
            val = constraints.get(field)
            if val is None:
                return
            max_score += w
            if str(row.get(field, "")).lower() == str(val).lower():
                score += w

        match("category", 1.0)
        match("color", 1.0)
        match("size", 0.8)
        match("material", 0.8)

        keywords = constraints.get("keywords") or []
        if keywords:
            max_score += 1.0
            title = str(row.get("title", "")).lower()
            hits = sum(1 for k in keywords if k.lower() in title)
            score += min(1.0, hits / len(keywords))

        return 0.0 if max_score == 0 else min(1.0, score / max_score)


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser("Late fusion pipeline")
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument(
        "--db",
        default="data/reviews-output/product_ranking.sqlite"
    )
    args = parser.parse_args()

    pipe = LateFusionPipeline(
        PipelineConfig(catalog_db_path=args.db)
    )

    out = pipe.run(args.image, args.text)

    print("\n--- LATE FUSION OUTPUT ---")
    print("SQL:\n", out["sql"])
    print("Params:\n", out["sql_params"])
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
