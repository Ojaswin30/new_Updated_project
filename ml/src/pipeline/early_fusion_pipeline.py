from typing import Dict, Any
import os
import time

from ml.src.pipeline.constraint_parser import ConstraintParser
from ml.src.pipeline.query_builder import SQLQueryBuilder
from ml.src.vision.early_fusion_clip_inference import early_fusion_image_infer


class EarlyFusionPipeline:
    """
    TRUE Symbolic Early Fusion

    - Image proposes structured attributes
    - Text proposes structured constraints
    - One joint deterministic decision
    - Only SQL generation (no DB execution)
    - Produces fusion diagnostics
    """

    def __init__(self):
        self.parser = ConstraintParser()

        self.sql_builder = SQLQueryBuilder(
            table_products="products",
            table_ranking="product_ranking",
            join_ranking=False
        )

    # ------------------------------------------------
    # JOINT FUSION DECISION
    # ------------------------------------------------
    def _joint_decision(
        self,
        image_signal: Dict[str, Any],
        text_constraints: Dict[str, Any]
    ) -> Dict[str, Any]:

        final = {}

        img_cat = image_signal.get("category")
        img_color = image_signal.get("color")

        txt_cat = text_constraints.get("category")
        txt_color = text_constraints.get("color")

        # Category logic:
        # Image proposes, text must not contradict
        if img_cat and (not txt_cat or txt_cat == img_cat):
            final["category"] = img_cat
        else:
            final["category"] = None

        # Color logic:
        # Text can refine image
        final["color"] = txt_color or img_color

        # Always respect hard text constraints
        for k in ["size", "material", "price_min", "price_max", "keywords"]:
            final[k] = text_constraints.get(k)

        return final

    # ------------------------------------------------
    # STATISTICS
    # ------------------------------------------------
    def _compute_statistics(
        self,
        image_signal: Dict[str, Any],
        text_constraints: Dict[str, Any],
        final_constraints: Dict[str, Any],
        runtime_ms: float
    ) -> Dict[str, Any]:

        stats = {}

        stats["category_agreement"] = (
            image_signal.get("category")
            and text_constraints.get("category")
            and image_signal["category"] == text_constraints["category"]
        )

        stats["color_agreement"] = (
            image_signal.get("color")
            and text_constraints.get("color")
            and image_signal["color"] == text_constraints["color"]
        )

        stats["image_confidence"] = {
            "category_score": image_signal.get("category_score", 0.0),
            "color_score": image_signal.get("color_score", 0.0),
        }

        active_constraints = sum(
            1 for k, v in final_constraints.items()
            if v not in [None, [], ""]
        )

        stats["active_constraint_count"] = active_constraints

        if final_constraints.get("category") and final_constraints.get("color"):
            stats["fusion_quality"] = "strong"
        elif final_constraints.get("category") or final_constraints.get("color"):
            stats["fusion_quality"] = "moderate"
        else:
            stats["fusion_quality"] = "weak"

        stats["runtime_ms"] = round(runtime_ms, 2)

        return stats

    # ------------------------------------------------
    # MAIN ENTRY
    # ------------------------------------------------
    def run(self, image_path: str, text: str) -> Dict[str, Any]:
        print("Running image inference...")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        start = time.time()

        # -------- Image Signal --------
        image_pred = early_fusion_image_infer(image_path)

        image_signal = {
            "category": image_pred["category"]
                if image_pred["category_score"] >= 0.25 else None,
            "color": image_pred["color"]
                if image_pred["color_score"] >= 0.35 else None,
            "category_score": image_pred["category_score"],
            "color_score": image_pred["color_score"]
        }

        # -------- Text Signal --------
        text_constraints = self.parser.parse(text)

        # -------- Early Fusion --------
        final_constraints = self._joint_decision(
            image_signal=image_signal,
            text_constraints=text_constraints
        )

        # -------- SQL Generation --------
        query = self.sql_builder.build(
            constraints=final_constraints,
            limit=200,
            sort_by="rating_desc"
        )

        runtime_ms = (time.time() - start) * 1000

        stats = self._compute_statistics(
            image_signal=image_signal,
            text_constraints=text_constraints,
            final_constraints=final_constraints,
            runtime_ms=runtime_ms
        )

        return {
            "fusion_mode": "symbolic_early",
            "image_signal": image_signal,
            "text_signal": text_constraints,
            "final_constraints": final_constraints,
            "query": {
                "sql": query.sql,
                "params": query.params
            },
            "statistics": stats
        }