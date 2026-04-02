from typing import Dict, Any, List
import os, time, sqlite3

from src.pipeline.constraint_parser import ConstraintParser
from src.pipeline.query_builder import SimpleQueryBuilder
from src.vision.early_fusion_clip_inference import early_fusion_image_infer

# ← Update this path to where you downloaded the SQLite file
DB_PATH = "D:\\github\\git repositories\\new_Updated_project\\ml\\data\\reviews-output\\product_ranking.sqlite"

#  In early_fusion_pipeline.py, add this mapping
CLIP_TO_DB_CATEGORY = {

    # -------------------- TOPS --------------------
    "t-shirt": "Tops and T-Shirts",
    "tee": "Tops and T-Shirts",
    "shirt": "Tops and T-Shirts",
    "linen shirt": "Tops and T-Shirts",
    "polo": "Tops and T-Shirts",
    "tank top": "Tops and T-Shirts",
    "crop top": "Tops and T-Shirts",
    "blouse": "Tops and T-Shirts",

    # -------------------- DRESSES / SKIRTS --------------------
    "dress": "Dresses and Skirts",
    "gown": "Dresses and Skirts",
    "maxi dress": "Dresses and Skirts",
    "mini dress": "Dresses and Skirts",
    "skirt": "Dresses and Skirts",
    "pleated skirt": "Dresses and Skirts",

    # -------------------- OUTERWEAR --------------------
    "jacket": "Jackets and Coats",
    "coat": "Jackets and Coats",
    "blazer": "Jackets and Coats",
    "overcoat": "Jackets and Coats",
    "hoodie": "Hoodies and Sweatshirts",
    "sweatshirt": "Hoodies and Sweatshirts",

    # -------------------- BOTTOMS --------------------
    "jeans": "Pants and Jeans",
    "denim": "Pants and Jeans",
    "pants": "Pants and Jeans",
    "trousers": "Pants and Jeans",
    "chinos": "Pants and Jeans",
    "leggings": "Activewear and Leggings",
    "yoga pants": "Activewear and Leggings",

    # -------------------- SHORTS --------------------
    "shorts": "Shorts",
    "bermuda": "Shorts",

    # -------------------- SHOES --------------------
    "sneakers": "Sneakers and Athletic Shoes",
    "running shoes": "Sneakers and Athletic Shoes",
    "sports shoes": "Sneakers and Athletic Shoes",
    "trainers": "Sneakers and Athletic Shoes",
    "boots": "Boots",
    "ankle boots": "Boots",
    "sandals": "Sandals and Flip Flops",
    "flip flops": "Sandals and Flip Flops",
    "chappals": "Sandals and Flip Flops",
    "heels": "Heels and Pumps",
    "pumps": "Heels and Pumps",
    "loafers": "Loafers and Flats",
    "flats": "Loafers and Flats",
    "slippers": "Slippers",

    # -------------------- BAGS --------------------
    "handbag": "Handbags and Purses",
    "purse": "Handbags and Purses",
    "shoulder bag": "Handbags and Purses",
    "backpack": "Backpacks",
    "rucksack": "Backpacks",
    "wallet": "Wallets",
    "card holder": "Wallets",
    "tote bag": "Tote Bags",
    "shopping bag": "Tote Bags",
    "crossbody bag": "Crossbody Bags",
    "sling bag": "Crossbody Bags",
    "clutch": "Clutches",
    "waist bag": "Fanny Packs",
    "fanny pack": "Fanny Packs",
    "luggage": "Luggage",
    "suitcase": "Luggage",

    # -------------------- EYEWEAR --------------------
    "sunglasses": "Sunglasses",
    "shades": "Sunglasses",
    "aviators": "Sunglasses",
    "reading glasses": "Reading Glasses",
    "spectacles": "Prescription Frames",
    "eyeglasses": "Prescription Frames",
    "goggles": "Goggles",

    # -------------------- WATCHES --------------------
    "watch": "Analog Watches",
    "wrist watch": "Analog Watches",
    "smartwatch": "Smartwatches",
    "smart watch": "Smartwatches",
    "digital watch": "Digital Watches",
    "watch strap": "Watch Accessories",

    # -------------------- JEWELRY --------------------
    "necklace": "Necklaces and Pendants",
    "pendant": "Necklaces and Pendants",
    "chain": "Necklaces and Pendants",
    "ring": "Rings",
    "earrings": "Earrings",
    "stud": "Earrings",
    "hoops": "Earrings",
    "bracelet": "Bracelets and Bangles",
    "bangle": "Bracelets and Bangles",
    "anklet": "Anklets",
    "brooch": "Brooches and Pins",
    "cufflinks": "Cufflinks",
    "jewelry set": "Jewelry Sets",

    # -------------------- SWIMWEAR --------------------
    "swimsuit": "Swimwear and Bikinis",
    "bikini": "Swimwear and Bikinis",
    "swimwear": "Swimwear and Bikinis",

    # -------------------- INTIMATES --------------------
    "bra": "Underwear and Intimates",
    "panties": "Underwear and Intimates",
    "lingerie": "Underwear and Intimates",
    "boxers": "Underwear and Intimates",

    # -------------------- HOSIERY --------------------
    "socks": "Socks and Hosiery",
    "stockings": "Socks and Hosiery",
    "tights": "Socks and Hosiery",

    # -------------------- KNITWEAR --------------------
    "sweater": "Sweaters and Cardigans",
    "jumper": "Sweaters and Cardigans",
    "cardigan": "Sweaters and Cardigans",

    # -------------------- ACCESSORIES --------------------
    "scarf": "Scarves and Wraps",
    "shawl": "Scarves and Wraps",
    "dupatta": "Scarves and Wraps",
    "belt": "Belts",
    "hat": "Hats and Caps",
    "cap": "Hats and Caps",
    "beanie": "Hats and Caps",
    "gloves": "Gloves and Mittens",
    "mittens": "Gloves and Mittens",
    "hairband": "Hair Accessories",
    "hair clip": "Hair Accessories",
    "scrunchie": "Hair Accessories",

    # -------------------- BEAUTY --------------------
    "sunscreen": "Sunscreen and Beauty",
    "sunblock": "Sunscreen and Beauty",
    "skincare": "Sunscreen and Beauty",
}



class EarlyFusionPipeline:

    def __init__(self):
        self.parser     = ConstraintParser()
        self.sql_builder = SimpleQueryBuilder(
            table_products="products",
            table_ranking="product_ranking",
            join_ranking=True
        )
    def _map_clip_category(self, clip_category: str) -> str:
        if not clip_category:
            return None
        clip_lower = clip_category.lower()
        # Exact match first
        if clip_lower in CLIP_TO_DB_CATEGORY:
            return CLIP_TO_DB_CATEGORY[clip_lower]
        # Partial match fallback
        for clip_kw, db_cat in CLIP_TO_DB_CATEGORY.items():
            if clip_kw in clip_lower or clip_lower in clip_kw:
                return db_cat
        return None
    def _joint_decision(self, image_signal, text_constraints):
        final = {}
        img_color    = image_signal.get("color")
        txt_color    = text_constraints.get("color")
        final["color"] = txt_color or img_color

        # Use image category if confident enough
        final["category"] = self._map_clip_category(image_signal.get("category"))

        for k in ["size", "material", "price_min", "price_max", "keywords"]:
            final[k] = text_constraints.get(k)
        return final

    def _compute_statistics(self, image_signal, text_constraints,
                             final_constraints, runtime_ms):
        stats = {}
        stats["category_agreement"] = (
            image_signal.get("category") and text_constraints.get("category")
            and image_signal["category"] == text_constraints["category"]
        )
        stats["color_agreement"] = (
            image_signal.get("color") and text_constraints.get("color")
            and image_signal["color"] == text_constraints["color"]
        )
        stats["image_confidence"] = {
            "category_score": image_signal.get("category_score", 0.0),
            "color_score":    image_signal.get("color_score", 0.0),
        }
        active = sum(1 for v in final_constraints.values() if v not in [None, [], ""])
        stats["active_constraint_count"] = active
        if final_constraints.get("category") and final_constraints.get("color"):
            stats["fusion_quality"] = "strong"
        elif final_constraints.get("category") or final_constraints.get("color"):
            stats["fusion_quality"] = "moderate"
        else:
            stats["fusion_quality"] = "weak"
        stats["runtime_ms"] = round(runtime_ms, 2)
        return stats

    # ------------------------------------------------
    # EXECUTE SQL AGAINST DB
    # ------------------------------------------------
    def _execute_query(self, sql: str, params: Dict) -> List[Dict]:
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"DB not found: {DB_PATH}")

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row          # returns dict-like rows
        c = conn.cursor()
        c.execute(sql, params)
        rows = [dict(row) for row in c.fetchall()]
        conn.close()
        return rows

    # ------------------------------------------------
    # MAIN ENTRY
    # ------------------------------------------------
    def run(self, image_path: str, text: str) -> Dict[str, Any]:
        print("Running image inference...")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        start = time.time()

        image_pred   = early_fusion_image_infer(image_path)
        image_signal = {
            "category":       image_pred["category"] if image_pred["category_score"] >= 0.25 else None,
            "color":          image_pred["color"]    if image_pred["color_score"]    >= 0.20 else None,
            "category_score": image_pred["category_score"],
            "color_score":    image_pred["color_score"],
        }

        text_constraints  = self.parser.parse(text)
        final_constraints = self._joint_decision(image_signal, text_constraints)

        query   = self.sql_builder.build(final_constraints, limit=200, sort_by="rating_desc")
        results = self._execute_query(query.sql, query.params)   # ← actually runs it

        runtime_ms = (time.time() - start) * 1000
        stats      = self._compute_statistics(image_signal, text_constraints,
                                               final_constraints, runtime_ms)

        return {
            "fusion_mode":       "symbolic_early",
            "image_signal":      image_signal,
            "text_signal":       text_constraints,
            "final_constraints": final_constraints,
            "query": {
                "sql":    query.sql,
                "params": query.params,
            },
            "results": {
                "count": len(results),
                "items": results[:10],      # show top 10 in output
            },
            "statistics": stats,
        }


if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--text",  required=True)
    args = parser.parse_args()

    pipeline = EarlyFusionPipeline()
    result   = pipeline.run(image_path=args.image, text=args.text)

    print("\n" + "="*60)
    print("EARLY FUSION RESULT")
    print("="*60)
    print(json.dumps(result, indent=2))