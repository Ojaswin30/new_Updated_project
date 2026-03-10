from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class QueryBuildResult:
    sql: str
    params: Dict[str, object]
    debug: Dict[str, object]

class SimpleQueryBuilder:
    def __init__(
        self,
        table_products: str = "products",
        table_ranking: str = "product_ranking",
        join_ranking: bool = True,
    ):
        self.table_products = table_products
        self.table_ranking  = table_ranking
        self.join_ranking   = join_ranking

    def build(
        self,
        constraints: Dict,
        limit: int = 200,
        sort_by: Optional[str] = "rating_desc",
    ) -> QueryBuildResult:

        where_clauses: List[str] = ["1=1"]
        params: Dict[str, object] = {}

        # Category filter
        if constraints.get("category"):
            where_clauses.append("p.category = :category")
            params["category"] = constraints["category"]

        # Color filter (new column)
        # Replace the color filter section with:
        if constraints.get("color"):
            where_clauses.append("""(
                LOWER(p.color) LIKE LOWER(:color)
                OR (p.color IS NULL AND LOWER(p.title) LIKE LOWER(:color))
            )""")
            params["color"] = f"%{constraints['color']}%"

        # Material filter (new column)
        if constraints.get("material"):
            where_clauses.append("LOWER(p.material) LIKE LOWER(:material)")
            params["material"] = f"%{constraints['material']}%"

        # Price filters (new column)
        if constraints.get("price_min") is not None:
            where_clauses.append("(p.price IS NULL OR p.price >= :price_min)")
            params["price_min"] = constraints["price_min"]

        if constraints.get("price_max") is not None:
            where_clauses.append("(p.price IS NULL OR p.price <= :price_max)")
            params["price_max"] = constraints["price_max"]

        # Keyword search on title + features
        keywords = constraints.get("keywords") or []
        for i, kw in enumerate(keywords[:5]):
            key = f"kw{i}"
            where_clauses.append(f"(p.title LIKE :{key} OR p.features LIKE :{key})")
            params[key] = f"%{kw}%"
        
        # FROM / JOIN
        from_sql = f"FROM {self.table_products} p"
        if self.join_ranking:
            from_sql += f"\nLEFT JOIN {self.table_ranking} r ON p.parent_asin = r.parent_asin"

        # SELECT
        select_fields = [
            "p.parent_asin", "p.title", "p.category",
            "p.image_url", "p.store", "p.average_rating",
            "p.rating_number", "p.price", "p.color", "p.material",
        ]
        if self.join_ranking:
            select_fields += [
                "COALESCE(r.review_score, 0.0) AS review_score",
                "COALESCE(r.num_reviews, 0)    AS num_reviews",
                "COALESCE(r.avg_rating, 0.0)   AS avg_rating",
            ]

        select_sql = "SELECT " + ", ".join(select_fields)

        # In query_builder.py, change the ORDER BY line to:
        if sort_by == "rating_desc" and self.join_ranking:
            order_sql = "ORDER BY r.review_score DESC, p.average_rating DESC"
        else:
            order_sql = "ORDER BY p.average_rating DESC"

        sql = f"{select_sql}\n{from_sql}\nWHERE {' AND '.join(where_clauses)}\n{order_sql}\nLIMIT :limit"
        params["limit"] = int(limit)

        debug = {
            "sort_by":       sort_by,
            "filters_used":  [k for k, v in constraints.items() if v not in [None, [], ""]],
            "join_ranking":  self.join_ranking,
        }

        return QueryBuildResult(sql=sql, params=params, debug=debug)