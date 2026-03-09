# ml/src/vision/clip_artifacts.py

import os
import torch
from transformers import CLIPModel, CLIPProcessor

# Resolve paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "models", "clip_vit_b32")
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_clip_artifacts():
    """
    Load CLIP model and processor.
    Weights are cached locally in ml/models/clip_vit_b32/.
    """

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        cache_dir=MODEL_CACHE_DIR
    ).to(DEVICE)

    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32",
        cache_dir=MODEL_CACHE_DIR
    )

    model.eval()

    return model, processor, DEVICE
