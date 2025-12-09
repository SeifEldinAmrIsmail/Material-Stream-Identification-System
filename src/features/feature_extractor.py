from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def preprocess_image(img, size=(128, 128)):
    """Resize image to a fixed size."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def extract_color_hist(img, bins=(8, 8, 8)):
    """
    Simple handcrafted feature:
    HSV color histogram with 8x8x8 bins = 512-dim vector.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        bins,
        [0, 180, 0, 256, 0, 256],
    )

    cv2.normalize(hist, hist)
    return hist.flatten()


def extract_features_from_path(img_path):
    """Load image from disk and return a 1D feature vector."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Could not read image for features: {img_path}")
        return None

    img = preprocess_image(img)
    feat = extract_color_hist(img)
    return feat


def build_feature_dataset(csv_path, output_prefix):
    """
    Read a CSV with columns [path, label, class_name],
    build feature matrix X and label vector y,
    and save them as .npy files.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    features = []
    labels = []

    for _, row in df.iterrows():
        img_path = row["path"]
        label = row["label"]

        feat = extract_features_from_path(img_path)
        if feat is None:
            # skip unreadable images
            continue

        features.append(feat)
        labels.append(label)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(output_prefix) + "_X.npy", X)
    np.save(str(output_prefix) + "_y.npy", y)

    print(f"[INFO] Saved features to {output_prefix}_X.npy with shape {X.shape}")
    print(f"[INFO] Saved labels   to {output_prefix}_y.npy with shape {y.shape}")


if __name__ == "__main__":
    # 1) Training set: use augmented CSV
    build_feature_dataset(
        "data/processed/train_augmented.csv",
        "data/processed/train",
    )

    # 2) Validation set: use original val split (no augmentation)
    build_feature_dataset(
        "data/interim/val_split.csv",
        "data/processed/val",
    )
