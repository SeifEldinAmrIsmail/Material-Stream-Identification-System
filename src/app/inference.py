# src/app/inference.py
"""
Simple inference script for the Material-Stream-Identification-System.

- Loads the pretrained ResNet18 feature extractor (same as training).
- Loads the trained KNN / SVM classifiers with rejection (from models/).
- Classifies a single image path from the command line.

Usage (from project root):

  # Using SVM + rejection (default)
  python src/app/inference.py --image data/raw/dataset/paper/some_image.jpg --model svm

  # Using KNN + rejection
  python src/app/inference.py --image data/raw/dataset/paper/some_image.jpg --model knn
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Literal

import joblib
import numpy as np
from PIL import Image, ImageFile

import torch
from torch import nn
from torchvision import models, transforms

# ---------------------------------------------------------------------
# 1. Global settings
# ---------------------------------------------------------------------

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Labels 0–5 are real classes, 6 is the rejection "unknown" class
LABEL_TO_CLASS = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash",
    6: "unknown",
}
UNKNOWN_LABEL = 6


# ---------------------------------------------------------------------
# 2. CNN feature extractor (ResNet-18)
# ---------------------------------------------------------------------

def load_cnn_model() -> nn.Module:
    """
    Load ResNet-18 pretrained on ImageNet, with final FC replaced by Identity,
    so that the output is a 512-dim feature vector.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    model.to(DEVICE)
    return model


# Same preprocessing as used in training
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def safe_open_image(path: Path) -> Image.Image:
    """Open image safely and convert to RGB. Raise RuntimeError if unreadable."""
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        return img
    except Exception as e:  # defensive
        raise RuntimeError(f"Could not read image: {path} ({e})") from e


def extract_cnn_features(
    img_path: Path,
    model: nn.Module,
) -> np.ndarray:
    """
    Given an image path and a loaded CNN model, return a (1, 512) numpy feature vector.
    """
    img = safe_open_image(img_path)
    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)

    with torch.no_grad():
        feat = model(x)  # (1, 512)

    feat_np = feat.cpu().numpy()  # (1, 512)
    return feat_np


# ---------------------------------------------------------------------
# 3. Load trained models (KNN + SVM bundles)
# ---------------------------------------------------------------------

def load_knn_bundle(models_dir: Path = Path("models")):
    """
    Load KNN model bundle from disk.

    Expected keys in the joblib file:
      - "scaler": fitted StandardScaler
      - "knn": fitted KNeighborsClassifier
      - "distance_threshold": float
      - "distance_percentile": int (for info only)
    """
    bundle_path = models_dir / "knn_cnn_with_rejection.pkl"
    if not bundle_path.exists():
        raise FileNotFoundError(f"KNN model bundle not found at {bundle_path}")
    bundle = joblib.load(bundle_path)
    return bundle


def load_svm_bundle(models_dir: Path = Path("models")):
    """
    Load SVM model bundle from disk.

    Expected keys in the joblib file:
      - "model": sklearn Pipeline (StandardScaler + SVC(probability=True))
      - "threshold": float
      - "label_to_class": dict (optional, but should match LABEL_TO_CLASS above)
      - "unknown_label": int (optional, default=6)
    """
    bundle_path = models_dir / "svm_cnn_with_rejection.pkl"
    if not bundle_path.exists():
        raise FileNotFoundError(f"SVM model bundle not found at {bundle_path}")
    bundle = joblib.load(bundle_path)
    return bundle


# ---------------------------------------------------------------------
# 4. Prediction helpers (always WITH rejection)
# ---------------------------------------------------------------------

def predict_knn_with_rejection(
    features: np.ndarray,
    bundle,
) -> Tuple[int, str, float, float]:
    """
    Predict using KNN + distance-based rejection.

    Args:
        features: numpy array of shape (1, 512) from the CNN
        bundle:   joblib bundle from load_knn_bundle()

    Returns:
        pred_label_id: int in {0..5, 6}
        pred_label_str: human-readable string
        distance: distance to nearest neighbor
        threshold: rejection distance threshold used
    """
    scaler = bundle["scaler"]
    knn = bundle["knn"]
    dist_threshold = float(bundle["distance_threshold"])

    # Scale feature
    X_scaled = scaler.transform(features)

    # Compute distance to nearest neighbor
    distances, _ = knn.kneighbors(X_scaled, n_neighbors=1)
    distance = float(distances[0, 0])

    # Normal KNN prediction
    pred = int(knn.predict(X_scaled)[0])

    # Apply rejection rule
    if distance > dist_threshold:
        pred = UNKNOWN_LABEL

    pred_str = LABEL_TO_CLASS.get(pred, "unknown")
    return pred, pred_str, distance, dist_threshold


def predict_svm_with_rejection(
    features: np.ndarray,
    bundle,
) -> Tuple[int, str, float, float]:
    """
    Predict using SVM + probability-based rejection.

    Args:
        features: numpy array of shape (1, 512) from the CNN
        bundle:   joblib bundle from load_svm_bundle()

    Returns:
        pred_label_id: int in {0..5, 6}
        pred_label_str: human-readable string
        max_proba: highest predicted class probability
        threshold: rejection probability threshold used
    """
    model = bundle["model"]
    threshold = float(bundle["threshold"])
    unknown_label = bundle.get("unknown_label", UNKNOWN_LABEL)

    # Predict probabilities
    proba = model.predict_proba(features)[0]  # (6,)
    max_proba = float(proba.max())
    pred = int(proba.argmax())

    # Rejection rule
    if max_proba < threshold:
        pred = unknown_label

    pred_str = LABEL_TO_CLASS.get(pred, "unknown")
    return pred, pred_str, max_proba, threshold


# ---------------------------------------------------------------------
# 5. High-level API and CLI
# ---------------------------------------------------------------------

def classify_image(
    image_path: str,
    model_name: Literal["svm", "knn"] = "svm",
) -> None:
    """
    End-to-end classification for a single image.
    Prints the result to the console.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # 1) Load CNN feature extractor (only once)
    cnn_model = load_cnn_model()

    # 2) Extract feature
    features = extract_cnn_features(img_path, cnn_model)

    # 3) Load chosen classifier & predict
    if model_name == "knn":
        bundle = load_knn_bundle()
        pred_id, pred_str, distance, threshold = predict_knn_with_rejection(
            features, bundle
        )
        print(f"\n[MODEL] k-NN (with rejection)")
        print(f"[IMAGE] {img_path}")
        print(f"[PRED]  class id = {pred_id}   ({pred_str})")
        print(f"[INFO]  nearest-neighbor distance = {distance:.3f}")
        print(f"[INFO]  rejection distance threshold = {threshold:.3f}")
        if pred_id == UNKNOWN_LABEL:
            print("[NOTE] Sample rejected as 'unknown' (too far from training data).")
    else:
        bundle = load_svm_bundle()
        pred_id, pred_str, max_proba, threshold = predict_svm_with_rejection(
            features, bundle
        )
        print(f"\n[MODEL] SVM (with rejection)")
        print(f"[IMAGE] {img_path}")
        print(f"[PRED]  class id = {pred_id}   ({pred_str})")
        print(f"[INFO]  max class probability = {max_proba:.3f}")
        print(f"[INFO]  rejection probability threshold = {threshold:.3f}")
        if pred_id == UNKNOWN_LABEL:
            print("[NOTE] Sample rejected as 'unknown' (low confidence).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify a single image with SVM or KNN (with rejection)."
    )
    parser.add_argument(
        "--image",
        "-i",
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=["svm", "knn"],
        default="svm",
        help="Which classifier to use (default: svm).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classify_image(args.image, model_name=args.model)


if __name__ == "__main__":
    main()
