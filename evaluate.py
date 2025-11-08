import os
import glob
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

DATA_ROOT = "dataset_processed"
OUT_DIR = "evaluation_outputs"
CONF_THRESH = 0.25
IOU_THRESH = 0.5
CLASS_NAMES = ["with_mask", "without_mask"]
os.makedirs(OUT_DIR, exist_ok=True)

def find_trained_weights():
    runs = sorted(glob.glob(os.path.join("runs", "detect", "*")))
    for run in runs:
        for candidate in ("weights/best.pt", "weights/last.pt"):
            p = os.path.join(run, candidate)
            if os.path.exists(p):
                return p
    return None


def draw_boxes_and_save(image_path, preds, out_path):
    img = cv2.imread(image_path)
    if img is None:
        return
    for p in preds:
        x1, y1, x2, y2, conf, cls = p
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label = (
            f"{CLASS_NAMES[int(cls)]} {conf:.2f}"
            if int(cls) < len(CLASS_NAMES)
            else f"{int(cls)} {conf:.2f}"
        )
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, max(10, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
    cv2.imwrite(out_path, img)


def plot_and_save_confusion_matrix(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main():
    weights = find_trained_weights()
    if weights:
        print("[INFO] Using trained weights:", weights)
        model_path = weights
    else:
        print("[WARN] No trained weights found. Using pretrained 'yolov8n.pt' for demo inference.")
        model_path = "yolov8n.pt"

    model = YOLO(model_path)

    val_imgs = sorted(glob.glob(os.path.join(DATA_ROOT, "images", "val", "*.jpg")))
    if not val_imgs:
        val_imgs = sorted(glob.glob(os.path.join(DATA_ROOT, "images", "valid", "*.jpg")))
    if not val_imgs:
        print("[ERROR] No validation images found.")
        return

    gt_list = []
    pred_list = []

    for img_path in val_imgs:
        base = Path(img_path).stem
        lbl_path = os.path.join(DATA_ROOT, "labels", "val", base + ".txt")
        if not os.path.exists(lbl_path):
            lbl_path = os.path.join(DATA_ROOT, "labels", "valid", base + ".txt")
        gt = []
        if os.path.exists(lbl_path):
            for L in open(lbl_path).read().splitlines():
                parts = L.strip().split()
                if len(parts) >= 5:
                    gt.append(int(parts[0]))

        res = model.predict(source=img_path, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)[0]
        preds = []
        if hasattr(res, "boxes") and len(res.boxes) > 0:
            for b in res.boxes:
                xyxy = b.xyxy[0].cpu().numpy()
                conf = float(b.conf[0].cpu().numpy())
                cls = int(b.cls[0].cpu().numpy())
                preds.append([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), conf, cls])

        if gt:
            for i, gcls in enumerate(gt):
                pred_cls = preds[i][5] if i < len(preds) else len(CLASS_NAMES)
                gt_list.append(gcls)
                pred_list.append(int(pred_cls))
        else:
            for p in preds:
                gt_list.append(len(CLASS_NAMES))
                pred_list.append(int(p[5]))

        out_file = os.path.join(OUT_DIR, base + "_pred.jpg")
        draw_boxes_and_save(img_path, preds, out_file)

    print("[INFO] Annotated output images saved to:", OUT_DIR)

    if len(gt_list) > 0:
        cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
        plot_and_save_confusion_matrix(
            np.array(gt_list, dtype=int),
            np.array(pred_list, dtype=int),
            CLASS_NAMES + ["no_detection"],
            cm_path,
        )
        print("[INFO] Confusion matrix saved to:", cm_path)
    else:
        print("[WARN] No data available to compute confusion matrix.")


if __name__ == "__main__":
    main()

