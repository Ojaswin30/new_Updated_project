from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List


# ----------------------------
# Utilities
# ----------------------------

def _normalize_text(text: str) -> str:
    """Lowercase + normalize spacing."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _load_vocab_file(filepath: str) -> List[str]:
    """Loads newline-separated vocabulary (ignores empty/comment lines)."""
    vocab: List[str] = []
    if not os.path.exists(filepath):
        return vocab

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if not line or line.startswith("#"):
                continue
            vocab.append(line)
    return vocab


def _find_best_match(text: str, vocab: List[str]) -> Optional[str]:
    """
    Returns the longest vocab item that appears in text (basic but effective).
    Ex: "light blue" should win over "blue" if both exist.
    """
    matches = [v for v in vocab if v in text]
    if not matches:
        return None
    matches.sort(key=len, reverse=True)
    return matches[0]


# ----------------------------
# Schema
# ----------------------------

@dataclass
class Constraints:
    category: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    material: Optional[str] = None
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    keywords: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        # ensure keywords defaults to []
        if d["keywords"] is None:
            d["keywords"] = []
        return d


# ----------------------------
# Main parser
# ----------------------------

class ConstraintParser:
    """
    Deterministic constraints parser:
    - category: matched from products vocab
    - color: matched from colors vocab
    - size: regex-based
    - price: regex-based (under/below, between, min/max)
    - material: small controlled vocab (extendable)
    """

    DEFAULT_MATERIALS = [
        "cotton", "polyester", "wool", "silk", "linen", "denim",
        "leather", "synthetic", "nylon", "rayon", "spandex",
        "viscose", "acrylic", "suede"
    ]

    def __init__(self, vocab_dir: Optional[str] = None):
        """
        vocab_dir should point to:
            ml/src/vision/vocab
        If not provided, it auto-resolves relative to this file location.
        """
        if vocab_dir is None:
            # Resolve: ml/src/pipeline/constraint_parser.py -> ml/src/vision/vocab
            current_dir = os.path.dirname(os.path.abspath(__file__))
            vocab_dir = os.path.normpath(os.path.join(current_dir, "..", "vision", "vocab"))

        self.vocab_dir = vocab_dir
        self.colors_vocab = _load_vocab_file(os.path.join(vocab_dir, "colors.txt"))
        self.products_vocab = _load_vocab_file(os.path.join(vocab_dir, "products.txt"))
        self.materials_vocab = self.DEFAULT_MATERIALS

    def parse(self, text: str) -> Dict:
        """
        Parse user query text into constraints dict.
        """
        normalized = _normalize_text(text)

        c = Constraints()
        c.keywords = []

        # 1) category (product type)
        c.category = _find_best_match(normalized, self.products_vocab)

        # 2) color
        c.color = _find_best_match(normalized, self.colors_vocab)

        # 3) size
        c.size = self._extract_size(normalized)

        # 4) material
        c.material = _find_best_match(normalized, self.materials_vocab)

        # 5) price range
        price_min, price_max = self._extract_price(normalized)
        c.price_min = price_min
        c.price_max = price_max

        # 6) keywords (remaining tokens after removing known items)
        c.keywords = self._extract_keywords(normalized, c)

        return c.to_dict()

    # ----------------------------
    # Extractors
    # ----------------------------

    def _extract_size(self, text: str) -> Optional[str]:
        """
        Extract apparel sizes (S/M/L/XL/XXL) and numeric waist sizes.
        Examples:
          - "size m"
          - "m size"
          - "size 32"
          - "waist 34"
        """
        # Letter sizes
        # Match: "size m", "m size", "size: xl", "xxl"
        letter_pat = r"\b(?:size\s*[:\-]?\s*)?(xxxl|xxl|xl|l|m|s|xs)\b"
        m = re.search(letter_pat, text)
        if m:
            return m.group(1).upper()

        # Numeric sizes like 28/30/32/34 etc
        num_pat = r"\b(?:size|waist)\s*[:\-]?\s*(\d{2})\b"
        m = re.search(num_pat, text)
        if m:
            return m.group(1)

        return None

    def _extract_price(self, text: str) -> tuple[Optional[int], Optional[int]]:
        """
        Extract price constraints.
        Handles:
          - "under 1500" / "below 2000" / "< 999"
          - "above 500" / "over 700" / "> 1000"
          - "between 500 and 1500" / "500-1500"
        Returns: (price_min, price_max)
        """
        # Normalize rupee signs / commas
        t = text.replace(",", "")
        t = t.replace("₹", "rs ").replace("inr", "rs ")

        # BETWEEN
        between_pat = r"\bbetween\s+(\d{2,7})\s+(?:and|to)\s+(\d{2,7})\b"
        m = re.search(between_pat, t)
        if m:
            lo = int(m.group(1))
            hi = int(m.group(2))
            return min(lo, hi), max(lo, hi)

        range_pat = r"\b(\d{2,7})\s*[-–]\s*(\d{2,7})\b"
        m = re.search(range_pat, t)
        if m:
            lo = int(m.group(1))
            hi = int(m.group(2))
            return min(lo, hi), max(lo, hi)

        # MAX
        max_pat = r"\b(?:under|below|less than|upto|up to|max)\s+(\d{2,7})\b"
        m = re.search(max_pat, t)
        if m:
            return None, int(m.group(1))

        max_pat_symbol = r"\b<\s*(\d{2,7})\b"
        m = re.search(max_pat_symbol, t)
        if m:
            return None, int(m.group(1))

        # MIN
        min_pat = r"\b(?:above|over|more than|min)\s+(\d{2,7})\b"
        m = re.search(min_pat, t)
        if m:
            return int(m.group(1)), None

        min_pat_symbol = r"\b>\s*(\d{2,7})\b"
        m = re.search(min_pat_symbol, t)
        if m:
            return int(m.group(1)), None

        return None, None

    def _extract_keywords(self, text: str, c: Constraints) -> List[str]:
        """
        Extract remaining keywords for search. This is intentionally conservative.
        Removes:
         - category
         - color
         - size tokens
         - material
         - price terms/numbers
        """
        t = text

        # Remove known fields
        for val in [c.category, c.color, c.material]:
            if val:
                t = t.replace(val.lower(), " ")

        # Remove size tokens
        if c.size:
            t = re.sub(rf"\b{re.escape(str(c.size).lower())}\b", " ", t)
            t = re.sub(r"\bsize\b", " ", t)
            t = re.sub(r"\bwaist\b", " ", t)

        # Remove price patterns
        t = re.sub(r"\b(under|below|less than|upto|up to|max|above|over|more than|min|between|and|to)\b", " ", t)
        t = re.sub(r"\b(rs|rupees|inr)\b", " ", t)
        t = re.sub(r"[<>₹]", " ", t)
        t = re.sub(r"\b\d{2,7}\b", " ", t)  # remove numbers

        # Tokenize
        tokens = re.findall(r"[a-z0-9]+", t.lower())
        # remove very short/noisy tokens
        stop = {"for", "with", "and", "or", "the", "a", "an", "of", "in", "on"}
        tokens = [tok for tok in tokens if tok not in stop and len(tok) >= 2]

        # de-duplicate while preserving order
        seen = set()
        out = []
        for tok in tokens:
            if tok in seen:
                continue
            seen.add(tok)
            out.append(tok)

        return out


# ----------------------------
# Convenience function
# ----------------------------

def parse_constraints(text: str) -> Dict:
    """
    Simple functional wrapper.
    """
    parser = ConstraintParser()
    return parser.parse(text)


if __name__ == "__main__":
    # quick local sanity test
    sample = "black tshirt size M cotton under 1500 oversized round neck"
    print(parse_constraints(sample))
