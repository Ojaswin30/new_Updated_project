import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_DIR = os.path.join(BASE_DIR, "vocab")

PRODUCTS_FILE = os.path.join(VOCAB_DIR, "products.txt")
COLORS_FILE = os.path.join(VOCAB_DIR, "colors.txt")


def _load_list(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing vocab file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_vocab():
    """
    Load controlled vocabulary once.
    """
    products = _load_list(PRODUCTS_FILE)
    colors = _load_list(COLORS_FILE)

    return products, colors
