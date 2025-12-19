from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def create_train_val_split(
    dataset_csv: str | Path = "data/interim/dataset.csv",
    train_csv: str | Path = "data/interim/train_split.csv",
    val_csv: str | Path = "data/interim/val_split.csv",
    val_ratio: float = 0.10,
    random_state: int = 42,
):
    df = pd.read_csv(dataset_csv)

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        stratify=df["label"],
        random_state=random_state,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_csv = Path(train_csv)
    val_csv = Path(val_csv)
    train_csv.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples  : {len(val_df)}")

    return train_df, val_df


if __name__ == "__main__":
    create_train_val_split()
