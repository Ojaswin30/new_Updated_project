"""
Simplified query builder for evaluation database
Only queries on fields that actually exist: parent_asin, title, category
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class QueryBuildResult:
    sql: str
    params: Dict[str, object]
    debug: Dict[str, object]

class SimpleQueryBuilder:
    """
    Query builder for evaluation database with minimal schema:
    - parent_asin, title, category, image_url, store, average_rating, rating_number
    
    Filters on category + title keywords only.
    Other constraints (color, size, material, price) will be matched in memory.
    """
    
    def __init__(
        self,
        table_products: str = "products",
        table_ranking: str = "product_ranking",
        join_ranking: bool = True,
    ):
        self.table_products = table_products
        self.table_ranking = table_ranking
        self.join_ranking = join_ranking

    def build(
        self,
        constraints: Dict,
        limit: int = 200,
        sort_by: Optional[str] = "relevance_desc",
    ) -> QueryBuildResult:
        """Build SQL query using only existing columns"""
        
        where_clauses: List[str] = ["1=1"]
        params: Dict[str, object] = {}

        # Category filter (exists in DB)
        if constraints.get("category"):
            where_clauses.append("p.category = :category")
            params["category"] = constraints["category"]

        # Keyword search on title
        keywords = constraints.get("keywords") or []
        keyword_clauses = []
        for i, kw in enumerate(keywords[:5]):
            key = f"kw{i}"
            keyword_clauses.append(f"(p.title LIKE :{key})")
            params[key] = f"%{kw}%"

        if keyword_clauses:
            where_clauses.append("(" + " AND ".join(keyword_clauses) + ")")

        # FROM / JOIN
        from_sql = f"FROM {self.table_products} p"
        if self.join_ranking:
            from_sql += f"""
LEFT JOIN {self.table_ranking} r
ON p.parent_asin = r.parent_asin
"""

        # SELECT
        select_fields = [
            "p.parent_asin",
            "p.title",
            "p.category",
            "p.image_url",
            "p.store",
            "p.average_rating",
            "p.rating_number",
        ]
        if self.join_ranking:
            select_fields += [
                "COALESCE(r.review_score, 0.0) AS review_score",
                "COALESCE(r.num_reviews, 0) AS num_reviews",
                "COALESCE(r.avg_rating, 0.0) AS avg_rating",
            ]

        select_sql = "SELECT " + ", ".join(select_fields)

        # ORDER BY
        if sort_by == "review_desc" and self.join_ranking:
            order_sql = "ORDER BY review_score DESC, p.average_rating DESC"
        else:
            order_sql = "ORDER BY p.average_rating DESC"

        # Final SQL
        sql = f"""
{select_sql}
{from_sql}
WHERE {" AND ".join(where_clauses)}
{order_sql}
LIMIT :limit
""".strip()

        params["limit"] = int(limit)

        debug = {
            "sort_by": sort_by,
            "filters": {"category": constraints.get("category")},
            "keywords_used": keywords[:5],
            "join_ranking": self.join_ranking,
        }

        return QueryBuildResult(sql=sql, params=params, debug=debug)