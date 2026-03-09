from typing import Dict, Any
import os

from ml.src.pipeline.constraint_parser import ConstraintParser
from ml.src.pipeline.query_builder import SQLQueryBuilder
from ml.src.vision.early_fusion_clip_inference import early_fusion_image_infer


class SymbolicEarlyFusionPipeline:
    """
    TRUE Symbolic Early Fusion

    - Image provides product anchor
    - Text refines / validates
    - ONE joint decision
    """

    def __init__(self):
        self.parser = ConstraintParser()
        self.sql_builder = SQLQueryBuilder(
            table_products="products",
            table_ranking="product_ranking",
            join_ranking=False
        )

    # ----------------------------
    # JOINT DECISION
    # ----------------------------
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

        # CATEGORY: image proposes, text may veto
        if img_cat and (not txt_cat or txt_cat == img_cat):
            final["category"] = img_cat
        else:
            final["category"] = None

        # COLOR: text allowed to refine
        final["color"] = txt_color or img_color

        # HARD TEXT CONSTRAINTS
        for k in ["size", "material", "price_min", "price_max", "keywords"]:
            final[k] = text_constraints.get(k)

        return final

    # ----------------------------
    # MAIN ENTRY
    # ----------------------------
    def run(self, image_path: str, text: str) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Image signal (FAST CLIP)
        image_pred = early_fusion_image_infer(image_path)
        image_signal = {
            "category": image_pred["category"]
                if image_pred["category_score"] >= 0.25 else None,
            "color": image_pred["color"]
                if image_pred["color_score"] >= 0.35 else None,
            "category_score": image_pred["category_score"],
            "color_score": image_pred["color_score"]
        }

        # Text signal
        text_constraints = self.parser.parse(text)

        # Joint early fusion
        final_constraints = self._joint_decision(
            image_signal=image_signal,
            text_constraints=text_constraints
        )

        # SQL projection
        query = self.sql_builder.build(
            constraints=final_constraints,
            limit=200,
            sort_by="rating_desc"
        )

        return {
            "image_signal": image_signal,
            "text_signal": text_constraints,
            "final_constraints": final_constraints,
            "query": {
                "sql": query.sql,
                "params": query.params
            }
        }
