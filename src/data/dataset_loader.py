from collections import Counter
from pathlib import Path
from typing import List, Tuple

import pandas as pd

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_TO_LABEL = {name: i for i, name in enumerate(CLASS_NAMES)}
LABEL_TO_CLASS = {i: name for name, i in CLASS_TO_LABEL.items()}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_image_paths_and_labels(
    dataset_root: str | Path,
) -> Tuple[List[Path], List[int], List[str]]:
    root = Path(dataset_root)
    paths, labels, class_names = [], [], []

    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        if not class_dir.exists():
            continue

        for p in class_dir.rglob("*"):
            if p.suffix.lower() in IMAGE_EXTS:
                paths.append(p)
                labels.append(CLASS_TO_LABEL[class_name])
                class_names.append(class_name)

    return paths, labels, class_names


def build_dataset_csv(
    dataset_root: str | Path = "data/raw/dataset",
    out_csv: str | Path = "data/interim/dataset.csv",
) -> pd.DataFrame:
    paths, labels, class_names = get_image_paths_and_labels(dataset_root)

    df = pd.DataFrame(
        {
            "path": [str(p) for p in paths],
            "label": labels,
            "class_name": class_names,
        }
    )

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Saved dataset CSV: {out_csv} ({len(df)} images)")

    counts = Counter(labels)
    for label, count in sorted(counts.items()):
        print(f"{LABEL_TO_CLASS[label]:10s}: {count}")

    return df


if __name__ == "__main__":
    build_dataset_csv()
