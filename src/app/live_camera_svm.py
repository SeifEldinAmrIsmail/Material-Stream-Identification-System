from pathlib import Path

import cv2
import joblib
from PIL import Image

import torch
from torch import nn
from torchvision import models, transforms


# ===============================
# 1) CNN FEATURE EXTRACTOR (ResNet-18)
# ===============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cnn_model():
    """
    Load pretrained ResNet-18 as a feature extractor (512-dim output).
    نفس الإعدادات المستخدمة في feature_extractor.py
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()  # remove final FC => 512-dim features
    model.eval()
    model.to(DEVICE)
    return model


# نفس الـ transform المستخدم في training
CNN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def image_to_feature(cnn_model, frame_bgr):
    """
    يأخذ frame من الكاميرا (BGR من OpenCV)،
    يحوله لـ 512-dim feature vector باستخدام ResNet-18.
    """
    # BGR -> RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    x = CNN_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)

    with torch.no_grad():
        feat = cnn_model(x)  # (1, 512)

    return feat.cpu().numpy()  # (1, 512)


# ===============================
# 2) LOAD SVM MODEL WITH REJECTION
# ===============================

def load_svm_bundle(models_dir="models"):
    """
    Load the trained SVM model bundle saved by train_svm.py
    (svm_cnn_with_rejection.pkl).
    """
    model_path = Path(models_dir) / "svm_cnn_with_rejection.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find {model_path}. "
            f"Make sure you ran src/models/train_svm.py first."
        )

    bundle = joblib.load(model_path)
    svm_model = bundle["model"]
    threshold = bundle["threshold"]
    label_to_class = bundle["label_to_class"]
    unknown_label = bundle["unknown_label"]
    return svm_model, threshold, label_to_class, unknown_label


def svm_predict_frame(svm_model, threshold, feature, label_to_class, unknown_label):
    """
    اعمل prediction لــ frame واحد:
    - input: feature شكلها (1, 512)
    - output: label name + probability
    """
    proba = svm_model.predict_proba(feature)  # (1, 6)
    max_p = float(proba.max(axis=1)[0])
    pred_idx = int(proba.argmax(axis=1)[0])

    # تطبيق الـ rejection rule
    if max_p < threshold:
        pred_label = unknown_label
    else:
        pred_label = pred_idx

    class_name = label_to_class.get(pred_label, "unknown")
    return class_name, max_p, pred_label


# ===============================
# 3) LIVE CAMERA APP
# ===============================

def main():
    print("[INFO] Loading CNN feature extractor (ResNet-18)...")
    cnn_model = load_cnn_model()

    print("[INFO] Loading SVM model with rejection...")
    svm_model, threshold, label_to_class, unknown_label = load_svm_bundle()

    print(f"[INFO] Using SVM rejection threshold = {threshold:.2f}")
    print("[INFO] Starting camera... (press 'q' to quit)")

    cap = cv2.VideoCapture(0)  # لو عندك أكتر من كاميرا استخدم 1 أو 2 بدلاً من 0
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame from camera.")
            break

        # 1) Extract CNN feature
        feature = image_to_feature(cnn_model, frame)  # (1, 512)

        # 2) SVM prediction + rejection
        class_name, max_p, pred_label = svm_predict_frame(
            svm_model, threshold, feature, label_to_class, unknown_label
        )

        # 3) Prepare label text
        if class_name == "unknown":
            display_text = f"Unknown / Not confident ({max_p:.2f})"
            color = (0, 0, 255)  # red
        else:
            display_text = f"{class_name} ({max_p:.2f})"
            color = (0, 255, 0)  # green

        # 4) Draw label on the frame
        cv2.putText(
            frame,
            display_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )

        # 5) Show frame
        cv2.imshow("Material Stream Classifier (SVM + CNN)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera app closed.")


if __name__ == "__main__":
    main()
