from ultralytics import YOLO
import os

DATA_YAML = "dataset_processed/data.yaml"   
MODEL = "yolov8n.pt"     
EPOCHS = 2              
BATCH = 2            
IMGSZ = 640
PROJECT = "face_mask_yolo"
NAME = "run1"

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


