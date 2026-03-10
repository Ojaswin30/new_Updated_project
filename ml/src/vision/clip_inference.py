from typing import Dict, Optional
from PIL import Image
import torch

from src.vision.clip_artifacts import load_clip_artifacts
from src.vision.clip_vocab import load_vocab


# ============================
# Load artifacts once
# ============================

_model, _processor, _device = load_clip_artifacts()
_PRODUCTS, _COLORS = load_vocab()


# ============================
# Prompt construction
# ============================

def _build_prompts():
    """
    Build prompt combinations dynamically.
    Example:
      - a photo of a tshirt
      - a photo of a black tshirt
    """
    prompts = []
    for product in _PRODUCTS:
        prompts.append(f"a photo of a {product}")
        for color in _COLORS:
            prompts.append(f"a photo of a {color} {product}")
    return prompts


_PROMPTS = _build_prompts()


# ============================
# Prompt parsing
# ============================

def _parse_prompt_label(label: str) -> Dict[str, Optional[str]]:
    """
    Parse prompt text into structured fields.
    Example:
      "a photo of a black tshirt"
        -> { "category": "tshirt", "color": "black" }
    """
    label = label.lower().replace("a photo of a", "").strip()
    tokens = label.split()

    if len(tokens) >= 2:
        color = tokens[0]
        product = " ".join(tokens[1:])
        if color in _COLORS:
            return {"category": product, "color": color}

    return {"category": label, "color": None}


# ============================
# Public API
# ============================

def classify_image(image_path: str, top_k: int = 3) -> Dict:
    """
    Classify an image using CLIP and return a normalized output
    compatible with late_fusion.py.

    Returns:
      {
        "category": str,
        "color": Optional[str],
        "score": float
      }
    """
    image = Image.open(image_path).convert("RGB")

    inputs = _processor(
        text=_PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(_device)

    with torch.no_grad():
        outputs = _model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)[0]
    topk = torch.topk(probs, k=top_k)

    best_idx = int(topk.indices[0])
    best_score = float(topk.values[0])
    best_label = _PROMPTS[best_idx]

    parsed = _parse_prompt_label(best_label)

    return {
        "category": parsed["category"],
        "color": parsed["color"],
        "score": best_score
    }


# ============================
# Late Fusion Wrapper
# ============================

class ClipInference:
    """
    Thin wrapper so late_fusion.py can call:
        clip.predict(image_path)
    """

    def predict(self, image_path: str) -> Dict:
        return classify_image(image_path, top_k=1)
