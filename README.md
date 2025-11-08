# 😷 **Face Mask Detection** — YOLOv8 + EfficientNet_B0 + Flask  

> 🧠 A complete end-to-end computer vision pipeline to detect whether a person is wearing a mask or not — from dataset to deployment!

---

## 🗂️ **Project Journey (My Step-by-Step Build 🚀)**

This is how I went from *zero → running web app*, using YOLOv8 + EfficientNet_B0 + Flask.  

---

### 🧩 **Step 1 — Choosing the Dataset**

I picked a dataset from **[Roboflow Universe](https://universe.roboflow.com)** called  
📦 `Face-mask-vsxay` under workspace `detection-and-segmentation`.  

It already had two classes:
- 😷 **with_mask**  
- 🙅‍♂️ **without_mask**

📥 **Dataset Download Code:**
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("detection-and-segmentation").project("face-mask-vsxay")
dataset = project.version(2).download("yolov8")
print("✅ Dataset downloaded successfully!")
```

📁 This creates a folder like:
```
Face-mask-2/
├── images/
└── labels/
```

---

### 🧹 **Step 2 — Preprocessing the Dataset**

To make the data YOLO-ready, I ran a custom script `preprocess.py` which:

🧺 **Cleans & Standardizes Data**
- Resizes all images → `640×640`
- Splits data into:
  - 🏋️ Train → 80%  
  - 🧪 Validation → 15%  
  - 🧫 Test → 5%
- Applies random horizontal flips (data augmentation)
- Generates `data.yaml` for YOLO

⚙️ **Run Command**
```bash
python preprocess.py
```

📂 **Output Structure**
```
dataset_processed/
├── images/train
├── images/val
├── images/test
├── labels/train
├── labels/val
├── labels/test
└── data.yaml
```

🧾 **Sample `data.yaml`**
```yaml
train: images/train
val: images/val
test: images/test
names:
  0: with_mask
  1: without_mask
```

---

### ⚙️ **Step 3 — Training the Detection Model (YOLOv8)**

Used **YOLOv8n (Nano)** — lightweight, fast, perfect for CPU/GPU.  

📜 **Training Script**
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="dataset_processed/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    project="face_mask_yolo",
    name="run1"
)
```

📦 **Results Saved in**
```
runs/detect/run1/
├── weights/
│   ├── best.pt
│   └── last.pt
└── results.png
```
💡 Use `best.pt` for evaluation or deployment.

---

### 🧠 **Step 4 — Classification Model (EfficientNet_B0)**

Even though YOLO detects masks directly, I added a **second-stage classifier** for confidence-boosting.  
🧍 Each detected face is cropped and classified as:

- 😷 `With Mask`  
- 🙅‍♂️ `Without Mask`

⚙️ **Model Setup**
```python
from torchvision import models
import torch.nn as nn

model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
```

🧑‍🏫 Trained for **10–20 epochs** using **Adam optimizer**  
💾 Saved as:
```
models/efficientnet_b0.pth
```

---

### 📊 **Step 5 — Model Evaluation**

📈 The script `evaluate.py` performs:
1. 🖼️ Annotated visual predictions (bounding boxes + labels)  
2. 📉 Confusion Matrix (via scikit-learn)

🧪 **Run Command**
```bash
python evaluate.py
```

📂 **Evaluation Output**
```
evaluation_outputs/
├── image1_pred.jpg
├── image2_pred.jpg
└── confusion_matrix.png
```

✅ The confusion matrix gives clear insights into YOLO’s accuracy and classification quality.

---

### 🌐 **Step 6 — Flask Web App Deployment**

💻 Final step — making everything interactive!  
The Flask app (`app.py`) integrates **YOLOv8** and **EfficientNet_B0** together.

🧩 **App Workflow**
1. 🖼️ Upload an image via web UI  
2. ⚙️ YOLO detects faces  
3. 🤖 Cropped faces sent to EfficientNet_B0  
4. 🎯 Final annotated image displayed instantly

🚀 **Run Command**
```bash
python app.py
```

🌍 Visit → [http://127.0.0.1:5000/](http://127.0.0.1:5000/)  
and test it live!

🖼️ **Demo Output Example:**  
![App Screenshot](Screenshot%202025-11-08%20224913.png)

---

### 🗃️ **Final Project Folder**

```
face-mask-detection/
├── app.py                     # Flask app for inference
├── download_roboflow.py       # Dataset download script
├── preprocess.py              # Preprocessing & augmentation
├── train_yolo.py              # YOLOv8 training
├── train_classifier.py        # EfficientNet_B0 training
├── evaluate.py                # Evaluation & visuals
│
├── dataset_processed/         # Clean dataset
├── models/                    # Trained weights
│   ├── yolov8_best.pt
│   └── efficientnet_b0.pth
├── runs/detect/run1/          # YOLO logs
├── evaluation_outputs/        # Confusion matrix & visuals
└── static/uploads/            # Uploaded images (Flask)
```

---

### ✅ **Summary — Full Workflow Recap**

🧩 Picked **Roboflow dataset (Face-mask-vsxay)**  
⬇️ Downloaded it in **YOLOv8 format**  
🧹 Preprocessed, resized, and augmented images  
⚙️ Trained **YOLOv8** for detection  
🧠 Trained **EfficientNet_B0** for classification  
📊 Evaluated results visually + confusion matrix  
🌐 Deployed in **Flask**  
🎉 Tested locally with random face images  

---

### 🧡 **Tech Stack**

| Category | Tools Used |
|-----------|-------------|
| Detection | YOLOv8 (Ultralytics) |
| Classification | EfficientNet_B0 (TorchVision) |
| Web App | Flask |
| Dataset | Roboflow |
| Evaluation | scikit-learn, OpenCV |
| Language | Python 🐍 |

---

### 💬 **Future Improvements**
- 🚀 Add live webcam detection  
- ☁️ Deploy on Render / AWS / Hugging Face Spaces  
- 📱 Build a Streamlit dashboard  

---

### 👨‍💻 **Created by Bharat Sharma**
> B.Tech ECE @ AKGEC • Python Developer Intern @ RxAdvance  
> 💬 *“From dataset to deployment — all in one neat AI pipeline.”*
