'''from ultralytics import YOLO
import os

DATA_YAML = "dataset_processed/data.yaml"
MODEL = "yolov8n.pt"
EPOCHS = 2
BATCH = 4
IMGZ = 640
PROJECT = "face_mask_yolo"
NAME = "run1"
USE_WANDB = False   
if __name__ == "__main__":
    os.environ["ULTRALYTICS_RUN_NAME"] = NAME
    if USE_WANDB:
        os.environ["WANDB_PROJECT"] = PROJECT
    model = YOLO(MODEL)
    model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMGZ, batch=BATCH, project=PROJECT, name=NAME)
    print("Training finished. Check outputs in ./runs/detect or wandb if enabled.")'''



"""
Very small wrapper to train a YOLOv8 model (beginner friendly).

Requirements: ultralytics (pip install ultralytics)
Run:
    python train_simple.py
"""

from ultralytics import YOLO
import os

# -------- CONFIG ----------
DATA_YAML = "dataset_processed/data.yaml"   # produced by preprocess_simple.py
MODEL = "yolov8n.pt"     # small model; change to a bigger one only if you understand GPU requirements
EPOCHS = 2              # increase if you want longer training
BATCH = 2             # reduce if out-of-memory (CPU/low-GPU)
IMGSZ = 640
PROJECT = "face_mask_yolo"
NAME = "run1"
# -------------------------

os.environ["ULTRALYTICS_RUN_NAME"] = NAME

def main():
    if not os.path.exists(DATA_YAML):
        print("[ERROR] data.yaml not found at", DATA_YAML)
        return
    print("[INFO] Loading model:", MODEL)
    model = YOLO(MODEL)
    print("[INFO] Starting training — this can take a while.")
    model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, project=PROJECT, name=NAME)
    print("[INFO] Training finished. Check runs/detect/<name>/ for outputs")

if __name__ == "__main__":
    main()

