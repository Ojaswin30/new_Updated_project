"""
Debug: Check if visual scores are actually varying
"""
import sqlite3
from ml.src.pipeline.constraint_parser import ConstraintParser

def compute_visual_importance(constraints: dict) -> float:
    score = 0.5
    if constraints.get('color'):
        score += 0.3
    if constraints.get('category'):
        score += 0.1
    if constraints.get('size'):
        score -= 0.2
    if constraints.get('material'):
        score -= 0.15
    if constraints.get('price_min') or constraints.get('price_max'):
        score -= 0.2
    keywords = constraints.get('keywords', [])
    if len(keywords) >= 3:
        score -= 0.15
    return max(0.0, min(1.0, score))

def compute_product_visual_match(query_constraints: dict, product_title: str) -> float:
    score = 0.5
    title_lower = product_title.lower()
    
    query_color = query_constraints.get('color')
    if query_color:
        if query_color.lower() in title_lower:
            score += 0.4
        else:
            score -= 0.3
    
    query_category = query_constraints.get('category')
    if query_category:
        if query_category.lower() in title_lower:
            score += 0.2
    
    return max(0.0, min(1.0, score))

parser = ConstraintParser()
db = sqlite3.connect("data/reviews-output/product_ranking.sqlite")
db.row_factory = sqlite3.Row
c = db.cursor()

# Get some sample queries
c.execute("SELECT * FROM queries LIMIT 5")
queries = [dict(row) for row in c.fetchall()]

# Get some products
c.execute("SELECT parent_asin, title FROM products LIMIT 10")
products = [dict(row) for row in c.fetchall()]

print("TESTING VISUAL SCORE VARIATION\n")

for q in queries:
    query_text = q['query_text']
    constraints = parser.parse(query_text)
    visual_imp = compute_visual_importance(constraints)
    
    print(f"Query: '{query_text}'")
    print(f"  Constraints: {constraints}")
    print(f"  Visual importance: {visual_imp:.3f}")
    print(f"  Product visual matches:")
    
    for p in products[:3]:
        v_match = compute_product_visual_match(constraints, p['title'])
        print(f"    - {p['title'][:50]}... → {v_match:.3f}")
    print()

db.close()