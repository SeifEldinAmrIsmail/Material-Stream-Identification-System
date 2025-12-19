from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd


def random_augment(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    ops = ["flip", "rotate", "bright", "contrast", "blur", "zoom"]

    # نختار 4 عمليات مختلفة عشوائياً
    n_ops = min(4, len(ops))
    for op in random.sample(ops, n_ops):
        if op == "flip":
            image = cv2.flip(image, 1)

        elif op == "rotate":
            angle = random.choice([-15, -10, -5, 5, 10, 15])
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(
                image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101
            )

        elif op == "bright":
            factor = random.uniform(0.7, 1.3)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

        elif op == "contrast":
            alpha = random.uniform(0.8, 1.4)
            beta = random.randint(-15, 15)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        elif op == "blur":
            image = cv2.GaussianBlur(image, (3, 3), 0)

        elif op == "zoom":
            zoom = random.uniform(0.9, 1.0)
            new_h, new_w = int(h * zoom), int(w * zoom)
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)
            crop = image[top : top + new_h, left : left + new_w]
            image = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)

    return image



def create_augmented_train(
    split_csv: str | Path = "data/interim/train_split.csv",
    output_root: str | Path = "data/processed/augmented_train",
    target_per_class: int = 1500,
):
    df = pd.read_csv(split_csv)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows = [df]

    for class_name, group in df.groupby("class_name"):
        original_paths = group["path"].tolist()
        class_label = int(group["label"].iloc[0])
        current_count = len(original_paths)

        target = max(target_per_class, current_count)
        num_to_add = target - current_count
        if num_to_add <= 0:
            continue

        print(f"[INFO] {class_name}: {current_count} -> {target} (adding {num_to_add})")

        class_dir = output_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        new_rows = []
        for i in range(num_to_add):
            src_path = Path(random.choice(original_paths))
            image = cv2.imread(str(src_path))
            if image is None:
                print(f"[WARN] Could not read image: {src_path}")
                continue

            aug_image = random_augment(image)
            dst_path = class_dir / f"{src_path.stem}_aug{i}.jpg"
            cv2.imwrite(str(dst_path), aug_image)

            new_rows.append(
                {
                    "path": dst_path.as_posix(),
                    "label": class_label,
                    "class_name": class_name,
                }
            )

        if new_rows:
            all_rows.append(pd.DataFrame(new_rows))

    full_df = pd.concat(all_rows, ignore_index=True)
    out_csv = Path("data/processed/train_augmented.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_csv, index=False)

    print("\n[SUMMARY]")
    print("Original train size:", len(df))
    print("Augmented train size:", len(full_df))


if __name__ == "__main__":
    create_augmented_train()
