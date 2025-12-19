from pathlib import Path

import cv2
import joblib
from PIL import Image

import torch
from torch import nn
from torchvision import models, transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def load_cnn_model():
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    model.to(DEVICE)
    return model


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def load_svm_bundle():
    bundle_path = Path("models") / "svm_cnn_with_rejection.pkl"
    bundle = joblib.load(bundle_path)
    return bundle["model"], bundle["threshold"]


def frame_to_feature(frame, cnn_model):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = cnn_model(x).cpu().numpy().squeeze()
    return feat.reshape(1, -1)


def predict_frame(frame, cnn_model, svm_model, threshold):
    feat = frame_to_feature(frame, cnn_model)
    proba = svm_model.predict_proba(feat)
    max_proba = proba.max(axis=1)[0]
    pred = proba.argmax(axis=1)[0]

    if max_proba < threshold:
        label_id = UNKNOWN_LABEL
    else:
        label_id = int(pred)

    label_name = LABEL_TO_CLASS.get(label_id, "unknown")
    return label_name, float(max_proba)


def main():
    print("[INFO] Loading CNN feature extractor...")
    cnn_model = load_cnn_model()

    print("[INFO] Loading SVM model...")
    svm_model, threshold = load_svm_bundle()
    print(f"[INFO] Using SVM rejection threshold = {threshold:.2f}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, score = predict_frame(frame, cnn_model, svm_model, threshold)

        text = f"{label} ({score:.2f})"

        if label == "unknown":
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Live Material Classifier (SVM)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
