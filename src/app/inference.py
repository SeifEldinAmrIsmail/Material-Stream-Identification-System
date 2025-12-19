from pathlib import Path
import argparse

import joblib
import numpy as np
from PIL import Image, ImageFile

import torch
from torch import nn
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def load_cnn_model():
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    model.to(DEVICE)
    return model


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def load_bundle(model_name: str):
    models_dir = Path("models")
    if model_name == "svm":
        path = models_dir / "svm_cnn_with_rejection.pkl"
    elif model_name == "knn":
        path = models_dir / "knn_cnn_with_rejection.pkl"
    else:
        raise ValueError("model_name must be 'svm' or 'knn'")
    return joblib.load(path)


def image_to_feature(image_path: str, cnn_model) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = cnn_model(x).cpu().numpy().squeeze()
    return feat.reshape(1, -1)


def predict_single_image(image_path: str, model_name: str = "svm"):
    cnn = load_cnn_model()
    bundle = load_bundle(model_name)
    feat = image_to_feature(image_path, cnn)

    if model_name == "svm":
        model = bundle["model"]
        threshold = bundle["threshold"]
        proba = model.predict_proba(feat)
        max_proba = proba.max(axis=1)
        preds = proba.argmax(axis=1)
        preds[max_proba < threshold] = UNKNOWN_LABEL
        score = float(max_proba[0])

    else:  # knn
        scaler = bundle["scaler"]
        knn = bundle["knn"]
        dist_th = bundle["distance_threshold"]
        feat_scaled = scaler.transform(feat)
        distances, _ = knn.kneighbors(feat_scaled, n_neighbors=1)
        distances = distances[:, 0]
        preds = knn.predict(feat_scaled)
        preds[distances > dist_th] = UNKNOWN_LABEL
        score = float(distances[0])

    label_id = int(preds[0])
    label_name = LABEL_TO_CLASS.get(label_id, "unknown")

    print(f"Image: {image_path}")
    print(f"Model: {model_name.upper()}")
    print(f"Predicted label: {label_id} ({label_name})")

    if model_name == "svm":
        print(f"Max probability: {score:.3f}")
    else:
        print(f"Nearest-neighbor distance: {score:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--model",
        choices=["svm", "knn"],
        default="svm",
        help="Which classifier bundle to use",
    )
    args = parser.parse_args()
    predict_single_image(args.image, args.model)


if __name__ == "__main__":
    main()
