from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
from torch import nn
from torchvision import models, transforms

# Allow PIL to load some truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    # Pretrained ResNet-18 on ImageNet
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    # Replace final FC layer with identity to get 512-dim features
    model.fc = nn.Identity()
    model.eval()
    model.to(DEVICE)
    return model


# Preprocessing expected by ResNet-18
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


def safe_open_image(path):
    """
    Try to open image with Pillow.
    Returns PIL.Image or None if completely unreadable.
    """
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"[WARN] Could not read image: {path} ({e})")
        return None


def extract_features_for_csv(csv_path, output_prefix):
    """
    Read CSV [path, label], extract 512-dim ResNet features
    for all *readable* images, and save X, y as .npy.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    paths = df["path"].tolist()
    labels = df["label"].tolist()

    model = load_model()

    feats_list = []
    labels_list = []

    for p, lbl in zip(paths, labels):
        img = safe_open_image(p)
        if img is None:
            # skip corrupted image
            continue

        x = transform(img).unsqueeze(0).to(DEVICE)  # shape (1, 3, 224, 224)

        with torch.no_grad():
            feat = model(x)           # shape (1, 512)
        feat = feat.cpu().numpy().squeeze()  # shape (512,)

        feats_list.append(feat)
        labels_list.append(int(lbl))

    X = np.stack(feats_list, axis=0)
    y = np.array(labels_list, dtype=np.int64)

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(output_prefix) + "_X.npy", X)
    np.save(str(output_prefix) + "_y.npy", y)

    print(f"[INFO] From CSV: {csv_path}")
    print(f"[INFO] Readable images: {X.shape[0]}")
    print(f"[INFO] Saved CNN features to {output_prefix}_X.npy with shape {X.shape}")
    print(f"[INFO] Saved labels       to {output_prefix}_y.npy with shape {y.shape}")


if __name__ == "__main__":
    # Training set: use augmented train
    extract_features_for_csv(
        "data/processed/train_augmented.csv",
        "data/processed/train_cnn",
    )

    # Validation set: original val split (no augmentation)
    extract_features_for_csv(
        "data/interim/val_split.csv",
        "data/processed/val_cnn",
    )
