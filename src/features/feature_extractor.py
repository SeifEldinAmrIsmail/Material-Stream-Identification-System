from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
from torch import nn
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
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


def safe_open_image(path: str):
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        print(f"[WARN] Could not read image: {path} ({e})")
        return None


def extract_features_for_csv(csv_path: str | Path, output_prefix: str | Path):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    paths = df["path"].tolist()
    labels = df["label"].tolist()

    model = load_model()
    feats, ys = [], []

    for p, lbl in zip(paths, labels):
        img = safe_open_image(p)
        if img is None:
            continue

        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feat = model(x).cpu().numpy().squeeze()

        feats.append(feat)
        ys.append(int(lbl))

    X = np.stack(feats, axis=0)
    y = np.array(ys, dtype=np.int64)

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(output_prefix) + "_X.npy", X)
    np.save(str(output_prefix) + "_y.npy", y)

    print(f"[INFO] {csv_path} -> {output_prefix}_X.npy {X.shape}")
    print(f"[INFO] {csv_path} -> {output_prefix}_y.npy {y.shape}")


if __name__ == "__main__":
    extract_features_for_csv(
        "data/processed/train_augmented.csv",
        "data/processed/train_cnn",
    )

    extract_features_for_csv(
        "data/interim/val_split.csv",
        "data/processed/val_cnn",
    )
