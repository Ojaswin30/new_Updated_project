"""
Debug: Does changing fusion weights actually change rankings?
"""
import sqlite3
from ml.src.pipeline.constraint_parser import ConstraintParser

def word_overlap_score(query: str, title: str) -> float:
    query_words = set(query.lower().split())
    title_words = set(title.lower().split())
    if not query_words:
        return 0.0
    return len(query_words & title_words) / len(query_words)

def compute_visual_match(constraints: dict, title: str) -> float:
    score = 0.3
    title_lower = title.lower()
    
    query_color = constraints.get('color')
    if query_color:
        if query_color.lower() in title_lower:
            return 0.95
        else:
            return 0.1
    
    query_category = constraints.get('category')
    if query_category:
        if query_category.lower() in title_lower:
            return 0.7
        else:
            return 0.2
    
    return score

parser = ConstraintParser()
db = sqlite3.connect("data/reviews-output/product_ranking.sqlite")
db.row_factory = sqlite3.Row
c = db.cursor()

# Get a visual query
query_text = "red dress"
constraints = parser.parse(query_text)

# Get products
c.execute("""
    SELECT p.parent_asin, p.title,
           COALESCE(r.review_score, 0.0) AS review_score
    FROM products p
    LEFT JOIN product_ranking r ON p.parent_asin = r.parent_asin
    LIMIT 20
""")

products = []
for row in c.fetchall():
    p = dict(row)
    p['text_score'] = word_overlap_score(query_text, p['title'])
    p['visual_score'] = compute_visual_match(constraints, p['title'])
    if p['text_score'] > 0:
        products.append(p)

print(f"Query: '{query_text}'")
print(f"Found {len(products)} products with text overlap\n")

# Show score distribution
print("Score distribution:")
for p in products[:5]:
    print(f"  {p['title'][:40]:40s} | V:{p['visual_score']:.2f} T:{p['text_score']:.2f} R:{p['review_score']:.2f}")

print("\n" + "="*80)
print("STATIC FUSION (α=0.50, β=0.35, γ=0.15)")
print("="*80)

static_ranked = []
for p in products:
    final = 0.50 * p['visual_score'] + 0.35 * p['text_score'] + 0.15 * p['review_score']
    static_ranked.append((p['title'], final, p['visual_score'], p['text_score'], p['review_score']))

static_ranked.sort(key=lambda x: x[1], reverse=True)

for i, (title, score, v, t, r) in enumerate(static_ranked[:5], 1):
    print(f"{i}. [{score:.3f}] {title[:50]}")
    print(f"   V={v:.2f} × 0.50 + T={t:.2f} × 0.35 + R={r:.2f} × 0.15")

print("\n" + "="*80)
print("DYNAMIC FUSION - VISUAL (α=0.75, β=0.10, γ=0.15)")
print("="*80)

dynamic_ranked = []
for p in products:
    final = 0.75 * p['visual_score'] + 0.10 * p['text_score'] + 0.15 * p['review_score']
    dynamic_ranked.append((p['title'], final, p['visual_score'], p['text_score'], p['review_score']))

dynamic_ranked.sort(key=lambda x: x[1], reverse=True)

for i, (title, score, v, t, r) in enumerate(dynamic_ranked[:5], 1):
    print(f"{i}. [{score:.3f}] {title[:50]}")
    print(f"   V={v:.2f} × 0.75 + T={t:.2f} × 0.10 + R={r:.2f} × 0.15")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

static_top5 = [x[0] for x in static_ranked[:5]]
dynamic_top5 = [x[0] for x in dynamic_ranked[:5]]

if static_top5 == dynamic_top5:
    print("❌ Rankings are IDENTICAL")
else:
    print("✅ Rankings are DIFFERENT")
    print(f"   {len(set(static_top5) & set(dynamic_top5))} products overlap in top-5")

db.close()