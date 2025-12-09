from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

CLASS_TO_LABEL: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
LABEL_TO_CLASS: Dict[int, str] = {idx: name for name, idx in CLASS_TO_LABEL.items()}

# allowed extensions (lowercase)
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_image_paths_and_labels(root_dir: str | Path) -> Tuple[List[Path], List[int]]:
    """
    Scan the dataset folder and return:
      - list of image paths
      - list of integer labels (same length as paths)
    Expected structure:
        root_dir/
            cardboard/
            glass/
            metal/
            paper/
            plastic/
            trash/
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    image_paths: List[Path] = []
    labels: List[int] = []

    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        if not class_dir.exists():
            print(f"[WARN] Class folder missing: {class_dir}")
            continue

        # iterate over files and filter by extension (case-insensitive)
        for path in class_dir.iterdir():
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTS:
                image_paths.append(path)
                labels.append(CLASS_TO_LABEL[class_name])

    return image_paths, labels


def describe_dataset(root_dir: str | Path) -> None:
    """
    Print how many images per class and total.
    """
    paths, labels = get_image_paths_and_labels(root_dir)
    counts = Counter(labels)

    print("Dataset summary:")
    for label_idx, count in counts.items():
        class_name = LABEL_TO_CLASS[label_idx]
        print(f"  {class_name:10s}: {count}")

    print(f"\nTotal images: {len(paths)}")


# Small self-test:
if __name__ == "__main__":
    # This assumes you run the script from the project root
    DATASET_DIR = "data/raw/dataset"
    describe_dataset(DATASET_DIR)
