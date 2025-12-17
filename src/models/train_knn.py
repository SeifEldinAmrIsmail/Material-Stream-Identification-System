# src/models/train_knn.py

from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Labels: 0–5 are real classes, 6 is "unknown" (rejected)
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

# Percentile to choose distance threshold for rejection (you can tune this)
KNN_DISTANCE_PERCENTILE = 95  # e.g. 95 → reject only the worst 5% distances


def knn_predict_with_rejection(knn, scaler, X, dist_threshold):
    """
    Predict with KNN and reject samples that are too far from the training data.
    - knn: fitted KNeighborsClassifier
    - scaler: fitted StandardScaler
    - X: raw feature matrix (N, d)
    - dist_threshold: distance above which we mark as 'unknown'
    """
    # Scale like training
    X_scaled = scaler.transform(X)

    # Distance to nearest neighbor for each sample
    distances, _ = knn.kneighbors(X_scaled, n_neighbors=1)
    distances = distances[:, 0]  # shape (N,)

    # Normal KNN predictions
    preds = knn.predict(X_scaled)

    # Copy and apply rejection rule
    preds_rej = preds.copy()
    preds_rej[distances > dist_threshold] = UNKNOWN_LABEL

    return preds_rej, distances


def main():
    # 1) Load CNN features
    X_train = np.load("data/processed/train_cnn_X.npy")
    y_train = np.load("data/processed/train_cnn_y.npy")
    X_val = np.load("data/processed/val_cnn_X.npy")
    y_val = np.load("data/processed/val_cnn_y.npy")

    print("Train shape:", X_train.shape, y_train.shape)
    print("Val shape  :", X_val.shape, y_val.shape)

    # 2) Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 3) Define & train KNN model
    knn = KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        metric="euclidean",
    )

    print("\n=== Training KNN on CNN features ===")
    knn.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # (A) NORMAL KNN – WITHOUT REJECTION
    # ------------------------------------------------------------------
    y_pred_plain = knn.predict(X_val_scaled)
    acc_plain = accuracy_score(y_val, y_pred_plain)
    print(f"\nKNN accuracy WITHOUT rejection: {acc_plain:.4f}")

    # ------------------------------------------------------------------
    # (B) KNN + REJECTION
    # ------------------------------------------------------------------

    # Use validation distances to choose a rejection threshold
    val_distances, _ = knn.kneighbors(X_val_scaled, n_neighbors=1)
    val_distances = val_distances[:, 0]

    dist_threshold = np.percentile(val_distances, KNN_DISTANCE_PERCENTILE)
    print(
        f"\nChosen KNN rejection distance threshold "
        f"(percentile {KNN_DISTANCE_PERCENTILE}): {dist_threshold:.3f}"
    )

    # Apply rejection rule
    y_pred_rej, distances = knn_predict_with_rejection(
        knn, scaler, X_val, dist_threshold
    )

    # Overall accuracy (unknown = counted as wrong, which is fine for reporting)
    acc_rej = accuracy_score(y_val, y_pred_rej)
    print(f"KNN accuracy WITH rejection: {acc_rej:.4f}")

    # Extra info: how many samples are rejected as unknown
    num_unknown = np.sum(y_pred_rej == UNKNOWN_LABEL)
    print(f"Number of rejected samples (predicted 'unknown'): {num_unknown} / {len(y_val)}")

    # ------------------------------------------------------------------
    # Detailed report for the "with rejection" predictions
    # ------------------------------------------------------------------
    labels_for_report = list(range(7))  # 0..6
    target_names = [LABEL_TO_CLASS[i] for i in labels_for_report]

    print("\nClassification report (including 'unknown' = 6):")
    print(
        classification_report(
            y_val,
            y_pred_rej,
            labels=labels_for_report,
            target_names=target_names,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_val,
        y_pred_rej,
        labels=labels_for_report,
    )
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # 4) Save model bundle (for inference)
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
