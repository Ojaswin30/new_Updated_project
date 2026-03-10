from typing import Dict, Optional
from PIL import Image
import torch

from src.vision.clip_artifacts import load_clip_artifacts
from src.vision.clip_vocab import load_vocab

# =========================================================
# Load CLIP ONCE (process lifetime)
# =========================================================

_model, _processor, _device = load_clip_artifacts()
_model.eval()
torch.set_grad_enabled(False)
torch.set_num_threads(1)

print(">>> EARLY FUSION CLIP LOADED <<<")

# =========================================================
# Controlled vocab (KEEP SMALL FOR SPEED)
# =========================================================

_PRODUCTS, _COLORS = load_vocab()

# OPTIONAL (HIGHLY RECOMMENDED FOR DEV)
# _PRODUCTS = ["bicycle", "tshirt", "shoe", "backpack"]
# _COLORS = ["black", "blue", "white", "red"]

# =========================================================
# Prompt construction
# =========================================================

_PRODUCT_PROMPTS = [f"a photo of a {p}" for p in _PRODUCTS]
_COLOR_PROMPTS = [f"a photo of a {c} object" for c in _COLORS]

# =========================================================
# Precompute TEXT embeddings ONCE
# =========================================================

with torch.no_grad():
    product_inputs = _processor(
        text=_PRODUCT_PROMPTS,
        return_tensors="pt",
        padding=True
    ).to(_device)

    color_inputs = _processor(
        text=_COLOR_PROMPTS,
        return_tensors="pt",
        padding=True
    ).to(_device)

    _PRODUCT_TEXT_FEATURES = _model.get_text_features(**product_inputs)
    _COLOR_TEXT_FEATURES = _model.get_text_features(**color_inputs)

    _PRODUCT_TEXT_FEATURES = torch.nn.functional.normalize(
        _PRODUCT_TEXT_FEATURES, dim=1
    )
    _COLOR_TEXT_FEATURES = torch.nn.functional.normalize(
        _COLOR_TEXT_FEATURES, dim=1
    )

# =========================================================
# Public API (FAST, IMAGE-ONLY)
# =========================================================

def early_fusion_image_infer(image_path: str) -> Dict[str, Optional[str]]:
    """
    Image-only CLIP inference for early fusion.

    Returns:
      {
        "category": str,
        "color": Optional[str],
        "category_score": float,
        "color_score": float
      }
    """

    image = Image.open(image_path).convert("RGB")

    image_inputs = _processor(
        images=image,
        return_tensors="pt"
    ).to(_device)

    with torch.no_grad():
        image_features = _model.get_image_features(**image_inputs)
        image_features = torch.nn.functional.normalize(
            image_features, dim=1
        )

        # Product
        product_sims = image_features @ _PRODUCT_TEXT_FEATURES.T
        prod_idx = int(product_sims.argmax())
        prod_score = float(product_sims[0, prod_idx])

        # Color
        color_sims = image_features @ _COLOR_TEXT_FEATURES.T
        color_idx = int(color_sims.argmax())
        color_score = float(color_sims[0, color_idx])

    return {
        "category": _PRODUCTS[prod_idx],
        "color": _COLORS[color_idx],
        "category_score": prod_score,
        "color_score": color_score
    }
