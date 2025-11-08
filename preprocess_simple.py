'''import os, glob, random
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2, yaml, shutil
import numpy as np
import albumentations as A
from tqdm import tqdm


RAW_ROOT = "Face-mask-2"   
OUT_ROOT = "dataset_processed"  
TARGET_IMG_SIZE = 640
VAL_RATIO = 0.15
TEST_RATIO = 0.05
AUG_PER_IMAGE = 1   
RANDOM_SEED = 42
CLASS_NAMES = ["with_mask", "without_mask"]


IMG_EXTS = [".jpg", ".jpeg", ".png"]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def collect_pairs(src):
    pairs = []
    for root, _, files in os.walk(src):
        for f in files:
            if Path(f).suffix.lower() in IMG_EXTS:
                img = os.path.join(root, f)
                lbl = os.path.splitext(img)[0] + ".txt"
                pairs.append((img, lbl))
    return pairs

def read_yolo(txt):
    if not os.path.exists(txt):
        return []
    lines = open(txt).read().strip().splitlines()
    out = []
    for L in lines:
        parts = L.split()
        if len(parts) >=5:
            out.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return out

def save_yolo(txt, labels):
    with open(txt, "w") as f:
        for lab in labels:
            f.write(f"{int(lab[0])} {lab[1]:.6f} {lab[2]:.6f} {lab[3]:.6f} {lab[4]:.6f}\n")

def make_aug_pipeline(size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.12, rotate_limit=12, p=0.6),
        A.GaussNoise(p=0.2),
        A.Resize(size, size)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'], min_visibility=0.2))

def ensure_dirs():
    for d in ["images/train","images/val","images/test","labels/train","labels/val","labels/test"]:
        os.makedirs(os.path.join(OUT_ROOT, d), exist_ok=True)

def process_item(img_path, labels, out_img, out_lbl, pipeline=None):
    img = cv2.imread(img_path)
    if img is None: 
        print("Unreadable:", img_path); return False
    if pipeline:
        bboxes = [(l[1], l[2], l[3], l[4]) for l in labels]
        cats = [l[0] for l in labels]
        transformed = pipeline(image=img, bboxes=bboxes, category_ids=cats)
        img_t = transformed["image"]
        boxes_t = transformed["bboxes"]
        cats_t = transformed["category_ids"]
        out_labs = [[cats_t[i], *boxes_t[i]] for i in range(len(boxes_t))]
    else:
        img_t = cv2.resize(img, (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
        out_labs = labels
    cv2.imwrite(out_img, img_t)
    save_yolo(out_lbl, out_labs)
    return True

def main():
    pairs = collect_pairs(RAW_ROOT)
    if not pairs:
        raise SystemExit("No images found in RAW_ROOT")
    print(f"Found {len(pairs)} images")

    has_train = any("/train/" in p[0].replace("\\","/") for p in pairs)
    if has_train:
        print("Existing splits detected — preserving.")
        grouped = {"train": [], "val": [], "test": []}
        for img,lbl in pairs:
            p = img.replace("\\","/")
            if "/train/" in p:
                grouped["train"].append((img,lbl))
            elif "/valid/" in p or "/val/" in p:
                grouped["val"].append((img,lbl))
            elif "/test/" in p:
                grouped["test"].append((img,lbl))
            else:
                grouped["train"].append((img,lbl))
    else:
        all_imgs = [p[0] for p in pairs]
        trainval, test = train_test_split(all_imgs, test_size=TEST_RATIO, random_state=RANDOM_SEED)
        train, val = train_test_split(trainval, test_size=VAL_RATIO/(1-TEST_RATIO), random_state=RANDOM_SEED)
        grouped = {"train": [(p, os.path.splitext(p)[0]+".txt") for p in train],
                   "val": [(p, os.path.splitext(p)[0]+".txt") for p in val],
                   "test": [(p, os.path.splitext(p)[0]+".txt") for p in test]}
        print(f"Created splits: train={len(grouped['train'])}, val={len(grouped['val'])}, test={len(grouped['test'])}")

    ensure_dirs()
    resize_pipeline = A.Compose([A.Resize(TARGET_IMG_SIZE, TARGET_IMG_SIZE)], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
    aug_pipeline = make_aug_pipeline(TARGET_IMG_SIZE)

    for split in ["train","val","test"]:
        out_img_dir = os.path.join(OUT_ROOT, "images", split)
        out_lbl_dir = os.path.join(OUT_ROOT, "labels", split)
        for img, lbl in tqdm(grouped[split], desc=f"Processing {split}"):
            labs = read_yolo(lbl)
            base = Path(img).stem
            process_item(img, labs, os.path.join(out_img_dir, base + ".jpg"), os.path.join(out_lbl_dir, base + ".txt"), pipeline=resize_pipeline)
            if split == "train" and AUG_PER_IMAGE>0:
                for ai in range(AUG_PER_IMAGE):
                    process_item(img, labs, os.path.join(out_img_dir, f"{base}_aug{ai}.jpg"), os.path.join(out_lbl_dir, f"{base}_aug{ai}.txt"), pipeline=aug_pipeline)

    data = {"path": os.path.abspath(OUT_ROOT), "train": "images/train", "val": "images/val", "test": "images/test", "names": CLASS_NAMES}
    with open(os.path.join(OUT_ROOT, "data.yaml"), "w") as f:
        yaml = __import__("yaml")
        yaml.safe_dump(data, f, sort_keys=False)
    print("Preprocessing done. Output:", OUT_ROOT)

if __name__ == "__main__":
    main()'''





"""
Simple preprocessing for YOLO dataset (beginner friendly).

What it does:
- Finds image files and matching .txt labels
- Creates train / val / test splits
- Resizes images to TARGET_IMG_SIZE
- Optionally creates a horizontally-flipped augmented copy for train images
- Writes dataset folder structure and a data.yaml for YOLO training

Usage:
    python preprocess_simple.py
"""

import os
import glob
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2
import random
import yaml

# --------- CONFIG (easy to change) ----------
RAW_ROOT = "Face-mask-2"        # folder where your downloaded Roboflow YOLO export lives
OUT_ROOT = "dataset_processed"  # output folder for processed images/labels
TARGET_IMG_SIZE = 640
VAL_RATIO = 0.15
TEST_RATIO = 0.05
AUG_FLIP = True                # create 1 flipped image per training image if True
RANDOM_SEED = 42
CLASS_NAMES = ["with_mask", "without_mask"]
IMG_EXTS = [".jpg", ".jpeg", ".png"]
# --------------------------------------------

random.seed(RANDOM_SEED)

def collect_image_label_pairs(src):
    """Return list of tuples (image_path, label_path). Label might not exist for an image."""
    pairs = []
    for ext in IMG_EXTS:
        for p in glob.glob(os.path.join(src, "**", f"*{ext}"), recursive=True):
            label = os.path.splitext(p)[0] + ".txt"
            pairs.append((p, label))
    return pairs

def ensure_dirs():
    for d in ["images/train","images/val","images/test","labels/train","labels/val","labels/test"]:
        os.makedirs(os.path.join(OUT_ROOT, d), exist_ok=True)

def resize_and_save_image(src_img_path, dest_img_path, size=TARGET_IMG_SIZE):
    img = cv2.imread(src_img_path)
    if img is None:
        print("[WARN] cannot read", src_img_path)
        return False
    resized = cv2.resize(img, (size, size))
    cv2.imwrite(dest_img_path, resized)
    return True

def copy_label(src_label_path, dest_label_path):
    if os.path.exists(src_label_path):
        shutil.copy(src_label_path, dest_label_path)
    else:
        # create empty label file if missing (YOLO expects file for each image)
        open(dest_label_path, "w").close()

def flip_image_and_label_horizontally(img_path, lbl_path, out_img_path, out_lbl_path):
    """Flip image horizontally and adjust YOLO-format labels accordingly.

    YOLO label format: class x_center y_center width height (all normalized).
    For horizontal flip, x_center -> 1 - x_center.
    """
    img = cv2.imread(img_path)
    if img is None:
        return False
    flipped = cv2.flip(img, 1)
    cv2.imwrite(out_img_path, flipped)

    if os.path.exists(lbl_path):
        lines = open(lbl_path).read().splitlines()
        out_lines = []
        for L in lines:
            parts = L.strip().split()
            if len(parts) >= 5:
                cls = parts[0]
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = parts[3]
                h = parts[4]
                x_center_new = 1.0 - x_center
                out_lines.append(f"{cls} {x_center_new:.6f} {y_center:.6f} {w} {h}")
        with open(out_lbl_path, "w") as f:
            f.write("\n".join(out_lines))
    else:
        open(out_lbl_path, "w").close()
    return True

def main():
    pairs = collect_image_label_pairs(RAW_ROOT)
    if not pairs:
        print("No images found in RAW_ROOT:", RAW_ROOT)
        return
    print(f"Found {len(pairs)} images")

    imgs = [p[0] for p in pairs]
    trainval, test = train_test_split(imgs, test_size=TEST_RATIO, random_state=RANDOM_SEED)
    train, val = train_test_split(trainval, test_size=VAL_RATIO/(1-TEST_RATIO), random_state=RANDOM_SEED)

    ensure_dirs()

    def process_split(split_list, split_name):
        for img_path in split_list:
            base = Path(img_path).stem
            src_label = os.path.splitext(img_path)[0] + ".txt"
            out_img = os.path.join(OUT_ROOT, "images", split_name, base + ".jpg")
            out_lbl = os.path.join(OUT_ROOT, "labels", split_name, base + ".txt")
            ok = resize_and_save_image(img_path, out_img)
            if not ok:
                continue
            copy_label(src_label, out_lbl)

            # augmentation only for train
            if split_name == "train" and AUG_FLIP:
                out_img_f = os.path.join(OUT_ROOT, "images", "train", base + "_flip.jpg")
                out_lbl_f = os.path.join(OUT_ROOT, "labels", "train", base + "_flip.txt")
                flip_image_and_label_horizontally(img_path, src_label, out_img_f, out_lbl_f)

    process_split(train, "train")
    process_split(val, "val")
    process_split(test, "test")

    # create data.yaml for YOLO training
    data = {
        "path": os.path.abspath(OUT_ROOT),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)}
    }
    with open(os.path.join(OUT_ROOT, "data.yaml"), "w") as f:
        yaml.dump(data, f)
    print("Preprocessing finished. Output folder:", OUT_ROOT)
    print("data.yaml written to", os.path.join(OUT_ROOT, "data.yaml"))

if __name__ == "__main__":
    main()


