from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from dataset_loader import get_image_paths_and_labels, LABEL_TO_CLASS

def create_and_save_splits(
    val_ratio=0.2,
    random_state=42,
):
    dataset_dir = Path("data/raw/dataset")
    interim_dir = Path("data/interim")
    interim_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load all image paths + labels
    paths, labels = get_image_paths_and_labels(dataset_dir)

    # 2) Stratified train/val split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths,
        labels,
        test_size=val_ratio,
        stratify=labels,
        random_state=random_state,
    )

    # 3) Helper to save one split
    def save_split(csv_name, paths_list, labels_list):
        df = pd.DataFrame({
            "path": [p.as_posix() for p in paths_list],
            "label": labels_list,
            "class_name": [LABEL_TO_CLASS[l] for l in labels_list],
        })
        csv_path = interim_dir / csv_name
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved {len(df)} samples to {csv_path}")

    # Save train and val
    save_split("train_split.csv", train_paths, train_labels)
    save_split("val_split.csv", val_paths, val_labels)

    print("\n[SUMMARY]")
    print("Train size:", len(train_paths))
    print("Val size  :", len(val_paths))


if __name__ == "__main__":
    create_and_save_splits()
