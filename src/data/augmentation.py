from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd


def random_augment(image):
    """
    Apply one simple random augmentation to an image.
    """
    choice = random.choice(["flip", "rotate", "bright"])

    if choice == "flip":
        # horizontal flip
        image = cv2.flip(image, 1)

    elif choice == "rotate":
        angle = random.choice([-15, -10, -5, 5, 10, 15])
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(
            image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101
        )

    elif choice == "bright":
        factor = random.uniform(0.7, 1.3)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)

    return image


def create_augmented_train(
    split_csv="data/interim/train_split.csv",
    output_root="data/processed/augmented_train",
    target_per_class=600,
):
    """
    Read the training split and create extra images to balance classes.
    - Saves augmented images under data/processed/augmented_train/<class_name>/
    - Creates data/processed/train_augmented.csv with original + augmented paths.
    """
    df = pd.read_csv(split_csv)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows = [df]  # start with original rows

    # group by class
    for class_name, group in df.groupby("class_name"):
        original_paths = group["path"].tolist()
        class_label = group["label"].iloc[0]
        current_count = len(original_paths)

        # we want at least target_per_class images per class
        target = max(target_per_class, current_count)
        num_to_add = target - current_count

        if num_to_add <= 0:
            print(f"[INFO] {class_name}: already has {current_count}, no augmentation needed.")
            continue

        print(f"[INFO] {class_name}: {current_count} -> {target} (adding {num_to_add} images)")

        class_dir = output_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        new_paths = []
        new_labels = []
        new_classes = []

        for i in range(num_to_add):
            src_path = Path(random.choice(original_paths))
            image = cv2.imread(str(src_path))

            if image is None:
                print(f"[WARN] Could not read image: {src_path}")
                continue

            aug_image = random_augment(image)
            new_name = f"{src_path.stem}_aug{i}.jpg"
            dst_path = class_dir / new_name

            cv2.imwrite(str(dst_path), aug_image)

            new_paths.append(dst_path.as_posix())
            new_labels.append(class_label)
            new_classes.append(class_name)

        aug_df = pd.DataFrame(
            {
                "path": new_paths,
                "label": new_labels,
                "class_name": new_classes,
            }
        )
        all_rows.append(aug_df)

    # combine original + augmented
    full_df = pd.concat(all_rows, ignore_index=True)
    out_csv = Path("data/processed/train_augmented.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_csv, index=False)

    print("\n[SUMMARY]")
    print("Original train size:", len(df))
    print("Augmented train size:", len(full_df))


if __name__ == "__main__":
    create_augmented_train()
