from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

LABEL_TO_CLASS = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash",
    6: "unknown",
}

UNKNOWN_LABEL = 6
KNN_DISTANCE_PERCENTILE = 95  # used to set rejection distance


def knn_predict_with_rejection(knn, scaler, X, dist_threshold):
    X_scaled = scaler.transform(X)
    distances, _ = knn.kneighbors(X_scaled, n_neighbors=1)
    distances = distances[:, 0]
    preds = knn.predict(X_scaled)
    preds[distances > dist_threshold] = UNKNOWN_LABEL
    return preds, distances


def main():
    X_train = np.load("data/processed/train_cnn_X.npy")
    y_train = np.load("data/processed/train_cnn_y.npy")
    X_val = np.load("data/processed/val_cnn_X.npy")
    y_val = np.load("data/processed/val_cnn_y.npy")

    print("Train shape:", X_train.shape, y_train.shape)
    print("Val shape  :", X_val.shape, y_val.shape)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    knn = KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        metric="euclidean",
    )

    print("\n=== Training KNN on CNN features ===")
    knn.fit(X_train_scaled, y_train)

    y_plain = knn.predict(X_val_scaled)
    acc_plain = accuracy_score(y_val, y_plain)
    print(f"\nKNN accuracy WITHOUT rejection: {acc_plain:.4f}")

    val_distances, _ = knn.kneighbors(X_val_scaled, n_neighbors=1)
    val_distances = val_distances[:, 0]
    dist_threshold = np.percentile(val_distances, KNN_DISTANCE_PERCENTILE)
    print(
        f"\nChosen KNN rejection distance threshold "
        f"(percentile {KNN_DISTANCE_PERCENTILE}): {dist_threshold:.3f}"
    )

    y_pred, distances = knn_predict_with_rejection(knn, scaler, X_val, dist_threshold)
    acc_rej = accuracy_score(y_val, y_pred)
    print(f"KNN accuracy WITH rejection: {acc_rej:.4f}")
    print(f"Rejected samples: {(y_pred == UNKNOWN_LABEL).sum()} / {len(y_val)}")

    labels = list(range(7))
    target_names = [LABEL_TO_CLASS[i] for i in labels]

    print("\nClassification report (WITH rejection):")
    print(
        classification_report(
            y_val,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_val, y_pred, labels=labels)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    bundle = {
        "scaler": scaler,
        "knn": knn,
        "distance_threshold": float(dist_threshold),
        "distance_percentile": KNN_DISTANCE_PERCENTILE,
    }
    model_path = models_dir / "knn_cnn_with_rejection.pkl"
    joblib.dump(bundle, model_path)
    print(f"\nSaved KNN model (with rejection info) to {model_path}")


if __name__ == "__main__":
    main()
