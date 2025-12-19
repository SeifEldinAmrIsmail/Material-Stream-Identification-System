from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
SVM_REJECTION_THRESHOLD = 0.50


def svm_predict_with_rejection(model, X, threshold):
    proba = model.predict_proba(X)
    max_proba = proba.max(axis=1)
    preds = proba.argmax(axis=1)
    preds[max_proba < threshold] = UNKNOWN_LABEL
    return preds, max_proba


def main():
    X_train = np.load("data/processed/train_cnn_X.npy")
    y_train = np.load("data/processed/train_cnn_y.npy")
    X_val = np.load("data/processed/val_cnn_X.npy")
    y_val = np.load("data/processed/val_cnn_y.npy")

    print("Train shape:", X_train.shape, y_train.shape)
    print("Val shape  :", X_val.shape, y_val.shape)

    svm_model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
    )

    print("\n=== Training SVM on CNN features ===")
    svm_model.fit(X_train, y_train)

    y_plain = svm_model.predict(X_val)
    acc_plain = accuracy_score(y_val, y_plain)
    print(f"\nSVM accuracy WITHOUT rejection: {acc_plain:.4f}")

    y_pred, max_proba = svm_predict_with_rejection(
        svm_model, X_val, SVM_REJECTION_THRESHOLD
    )
    acc_rej = accuracy_score(y_val, y_pred)
    print(f"SVM accuracy WITH rejection: {acc_rej:.4f}")
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
