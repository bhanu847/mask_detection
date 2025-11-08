import os
import glob
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2
import random
import yaml

RAW_ROOT = "Face-mask-2"        
OUT_ROOT = "dataset_processed"  
TARGET_IMG_SIZE = 640
VAL_RATIO = 0.15
TEST_RATIO = 0.05
AUG_FLIP = True                
RANDOM_SEED = 42
CLASS_NAMES = ["with_mask", "without_mask"]
IMG_EXTS = [".jpg", ".jpeg", ".png"]

random.seed(RANDOM_SEED)

def collect_image_label_pairs(src):
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
        open(dest_label_path, "w").close()

def flip_image_and_label_horizontally(img_path, lbl_path, out_img_path, out_lbl_path):
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

            if split_name == "train" and AUG_FLIP:
                out_img_f = os.path.join(OUT_ROOT, "images", "train", base + "_flip.jpg")
                out_lbl_f = os.path.join(OUT_ROOT, "labels", "train", base + "_flip.txt")
                flip_image_and_label_horizontally(img_path, src_label, out_img_f, out_lbl_f)

    process_split(train, "train")
    process_split(val, "val")
    process_split(test, "test")

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



