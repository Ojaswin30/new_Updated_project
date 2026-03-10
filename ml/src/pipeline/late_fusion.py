from __future__ import annotations

import argparse
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from src.pipeline.constraint_parser import ConstraintParser
from src.pipeline.query_builder import SimpleQueryBuilder as SQLQueryBuilder, QueryBuildResult
from src.vision.early_fusion_clip_inference import early_fusion_image_infer


# ----------------------------
# Data Contracts
# ----------------------------

@dataclass
class PipelineConfig:
    catalog_db_path: str = "data/reviews-output/product_ranking.sqlite"
    products_table:  str = "products"
    ranking_table:   str = "product_ranking"
    sql_limit:       int = 200
    top_k_return:    int = 20
    alpha_visual:    float = 0.50
    beta_constraints:float = 0.35
    gamma_review:    float = 0.15


# ----------------------------
# Category Mapping
# ----------------------------

CLIP_TO_DB_CATEGORY = {
    "linen shirt": "Tops",   "t-shirt": "Tops",      "dress": "Dresses",
    "jacket": "Outerwear",   "jeans": "Bottoms",      "sneakers": "Shoes",
    "handbag": "Bags",       "sunglasses": "Eyewear", "watch": "Watches",
    "necklace": "Jewelry",   "bracelet": "Jewelry",   "earrings": "Jewelry",
    "swimsuit": "Swimwear",  "hoodie": "Outerwear",   "skirt": "Dresses",
    "leggings": "Bottoms",   "boots": "Shoes",        "scarf": "Accessories",
    "belt": "Accessories",   "hat": "Accessories",    "socks": "Hosiery",
    "bra": "Intimates",      "sweater": "Knitwear",   "cardigan": "Knitwear",
}


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
        vision_out       = self._run_vision(image_path)
        text_constraints = self.parser.parse(text)
        merged           = self._merge_constraints(vision_out, text_constraints)

        query_result = self.sql_builder.build(
            constraints=merged,
            limit=self.config.sql_limit,
            sort_by="rating_desc",
        )

        candidates = self._execute_sql(query_result)
        ranked     = self._rank_candidates(candidates, merged)

        return {
            "constraints_text":   text_constraints,
            "constraints_vision": vision_out,
            "constraints_merged": merged,
            "sql":        query_result.sql,
            "sql_params": query_result.params,
            "sql_debug":  query_result.debug,
            "results":    ranked[: self.config.top_k_return],
        }

    # ----------------------------
    # Vision
    # ----------------------------

    def _run_vision(self, image_path: str) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_pred = early_fusion_image_infer(image_path)
        return {
            "category":       image_pred.get("category") if image_pred.get("category_score", 0) >= 0.25 else None,
            "color":          image_pred.get("color")    if image_pred.get("color_score", 0)    >= 0.20 else None,
            "category_score": image_pred.get("category_score", 0.0),
            "color_score":    image_pred.get("color_score", 0.0),
            "visual_score":   float(image_pred.get("category_score", 0.0)),
        }

    # ----------------------------
    # Constraint Merge
    # ----------------------------

    def _map_clip_category(self, clip_category: str) -> Optional[str]:
        if not clip_category:
            return None
        clip_lower = clip_category.lower()
        if clip_lower in CLIP_TO_DB_CATEGORY:
            return CLIP_TO_DB_CATEGORY[clip_lower]
        for kw, cat in CLIP_TO_DB_CATEGORY.items():
            if kw in clip_lower or clip_lower in kw:
                return cat
        return None

    def _merge_constraints(self, vision: Dict, text: Dict) -> Dict:
        merged = dict(text)
        if not merged.get("category") and vision.get("category"):
            merged["category"] = self._map_clip_category(vision["category"])
        if not merged.get("color") and vision.get("color"):
            merged["color"] = vision["color"]
        merged["_vision_score"] = vision.get("visual_score", 0.0)
        return merged

    # ----------------------------
    # SQL Execution
    # ----------------------------

    def _execute_sql(self, query: QueryBuildResult) -> List[Dict[str, Any]]:
        if not os.path.exists(self.config.catalog_db_path):
            raise FileNotFoundError(f"DB not found: {self.config.catalog_db_path}")

        conn = sqlite3.connect(self.config.catalog_db_path)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            cur.execute(query.sql, query.params)
            return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()

    # ----------------------------
    # Late Fusion Re-Ranking
    # ----------------------------

    def _constraint_match_score(self, row: Dict, constraints: Dict) -> float:
        score, max_score = 0.0, 0.0

        def fuzzy_match(field: str, w: float):
            nonlocal score, max_score
            val = constraints.get(field)
            if val is None:
                return
            max_score += w
            row_val = str(row.get(field) or "").lower()
            con_val = str(val).lower()
            if not row_val:
                score += 0.0            # missing = no contribution
            elif con_val in row_val:
                score += w              # e.g. "navy blue" contains "blue"
            elif row_val in con_val:
                score += w * 0.8        # reverse partial
            # else no match = 0

        fuzzy_match("category", 1.0)
        fuzzy_match("color",    1.0)
        fuzzy_match("size",     0.8)
        fuzzy_match("material", 0.8)

        # Price score — cheaper relative to max = better
        price_max = constraints.get("price_max")
        if price_max is not None:
            max_score += 0.5
            price = row.get("price")
            if price is not None:
                score += 0.5 * max(0.0, 1.0 - (price / price_max))
            else:
                score += 0.25           # NULL price = neutral

        # Keyword match on title
        keywords = constraints.get("keywords") or []
        if keywords:
            max_score += 1.0
            title = str(row.get("title", "")).lower()
            hits  = sum(1 for k in keywords if k.lower() in title)
            score += min(1.0, hits / len(keywords))

        return 0.0 if max_score == 0 else min(1.0, score / max_score)

    def _rank_candidates(self, candidates: List[Dict], constraints: Dict) -> List[Dict]:
        ranked = []
        for row in candidates:
            visual     = float(constraints.get("_vision_score", 0.0))
            constraint = self._constraint_match_score(row, constraints)
            review     = float(row.get("review_score", 0.0))

            final = (
                self.config.alpha_visual     * visual
                + self.config.beta_constraints * constraint
                + self.config.gamma_review     * review
            )

            ranked.append({
                "parent_asin":      row.get("parent_asin"),
                "title":            row.get("title"),
                "final_score":      round(final, 6),
                "visual_score":     round(visual, 6),
                "constraint_score": round(constraint, 6),
                "review_score":     round(review, 6),
                "price":            row.get("price"),
                "color":            row.get("color"),
                "category":         row.get("category"),
                "image_url":        row.get("image_url"),
            })

        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        return ranked


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser("Late fusion pipeline")
    parser.add_argument("--image", required=True)
    parser.add_argument("--text",  required=True)
    parser.add_argument("--db", default="data/reviews-output/product_ranking.sqlite")
    args = parser.parse_args()

    pipe = LateFusionPipeline(PipelineConfig(catalog_db_path=args.db))
    out  = pipe.run(args.image, args.text)

    print("\n--- LATE FUSION OUTPUT ---")
    print("SQL:\n", out["sql"])
    print("\nTop results:")
    for i, r in enumerate(out["results"], 1):
        print(
            f"{i:02d}. [{r['parent_asin']}] score={r['final_score']:.4f} "
            f"(visual={r['visual_score']:.3f}, "
            f"constraint={r['constraint_score']:.3f}, "
            f"review={r['review_score']:.3f}) "
            f"${r['price'] or 'N/A'} | {r['title']}"
        )


if __name__ == "__main__":
    main()