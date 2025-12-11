from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

LABEL_TO_CLASS = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash",
}


def main():
    # 1) Load features
    X_train = np.load("data/processed/train_X.npy")
    y_train = np.load("data/processed/train_y.npy")
    X_val = np.load("data/processed/val_X.npy")
    y_val = np.load("data/processed/val_y.npy")

    print("Train shape:", X_train.shape, y_train.shape)
    print("Val shape  :", X_val.shape, y_val.shape)

    # 2) Hyperparameter grid (small & simple)
    C_values = [0.1, 1.0, 10.0, 50.0]
    gamma_values = ["scale", 0.1, 0.01, 0.001]

    best_acc = 0.0
    best_params = None
    best_model = None

    print("\n=== Tuning SVM (C, gamma) ===")
    for C in C_values:
        for gamma in gamma_values:
            print(f"Trying C={C}, gamma={gamma} ...", end=" ")

            model = make_pipeline(
                StandardScaler(),
                SVC(
                    kernel="rbf",
                    C=C,
                    gamma=gamma,
                    class_weight="balanced",  # help classes like 'trash'
                    probability=True,
                    random_state=42,
                ),
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            print(f"val_acc={acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_params = (C, gamma)
                best_model = model

    print("\nBest validation accuracy:", best_acc)
    print("Best params: C =", best_params[0], ", gamma =", best_params[1])

    # 3) Save the best model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "svm_tuned_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"\nSaved tuned SVM model to {model_path}")


if __name__ == "__main__":
    main()
