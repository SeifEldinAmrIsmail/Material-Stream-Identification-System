# src/models/train_svm.py

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Labels 0–5 are real, 6 is the rejection class
LABEL_TO_CLASS = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash",
    6: "unknown",   # rejection class
}

UNKNOWN_LABEL = 6

# Fixed probability threshold for rejection (you can tune this)
SVM_REJECTION_THRESHOLD = 0.50


def svm_predict_with_rejection(model, X, threshold=SVM_REJECTION_THRESHOLD):
    """
    SVM prediction with rejection.

    - model: sklearn Pipeline (StandardScaler + SVC(probability=True))
    - X: feature matrix (N, d)
    returns:
      preds_rej: labels 0..5, or 6 for 'unknown'
      max_proba: max probability per sample
    """
    proba = model.predict_proba(X)        # shape (N, 6)
    max_proba = proba.max(axis=1)         # shape (N,)
    preds = proba.argmax(axis=1)          # predicted classes 0..5

    preds_rej = preds.copy()
    # low-confidence -> unknown (6)
    preds_rej[max_proba < threshold] = UNKNOWN_LABEL
    return preds_rej, max_proba


def main():
    # ------------------------------
    # 1) Load CNN features
    # ------------------------------
    X_train = np.load("data/processed/train_cnn_X.npy")
    y_train = np.load("data/processed/train_cnn_y.npy")
    X_val = np.load("data/processed/val_cnn_X.npy")
    y_val = np.load("data/processed/val_cnn_y.npy")

    print("Train shape:", X_train.shape, y_train.shape)
    print("Val shape  :", X_val.shape, y_val.shape)

    # ------------------------------
    # 2) Build & train SVM
    # ------------------------------
    svm_model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,   # needed for predict_proba
            random_state=42,
        ),
    )

    print("\n=== Training SVM on CNN features ===")
    svm_model.fit(X_train, y_train)

    # ----------------------------------------------------------
    # (A) NORMAL SVM – WITHOUT REJECTION
    # ----------------------------------------------------------
    y_pred_plain = svm_model.predict(X_val)
    acc_plain = accuracy_score(y_val, y_pred_plain)
    print(f"\nSVM accuracy WITHOUT rejection: {acc_plain:.4f}")

    # ----------------------------------------------------------
    # (B) SVM + REJECTION (probability-based)
    # ----------------------------------------------------------
    print(
        f"\nUsing SVM rejection probability threshold: "
        f"{SVM_REJECTION_THRESHOLD:.2f}"
    )

    y_pred_rej, max_proba = svm_predict_with_rejection(
        svm_model, X_val, threshold=SVM_REJECTION_THRESHOLD
    )
    acc_rej = accuracy_score(y_val, y_pred_rej)
    print(f"SVM accuracy WITH rejection: {acc_rej:.4f}")

    num_unknown = np.sum(y_pred_rej == UNKNOWN_LABEL)
    print(f"Number of rejected samples (predicted 'unknown'): {num_unknown} / {len(y_val)}")

    # ----------------------------------------------------------
    # Detailed report for the WITH-rejection predictions
    # ----------------------------------------------------------
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

    # ------------------------------
    # 4) Save model + threshold
    # ------------------------------
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    bundle = {
        "model": svm_model,
        "threshold": SVM_REJECTION_THRESHOLD,
        "label_to_class": LABEL_TO_CLASS,
        "unknown_label": UNKNOWN_LABEL,
    }

    model_path = models_dir / "svm_cnn_with_rejection.pkl"
    joblib.dump(bundle, model_path)
    print(f"\nSaved SVM model (with rejection info) to {model_path}")


if __name__ == "__main__":
    main()
