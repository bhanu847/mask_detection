🗂️ Project Structure (and My Step-by-Step Journey 🚀)

So here’s how I approached this entire face mask detection project from scratch — from picking a random dataset to running it live with Flask.

🧩 Step 1 — Choosing the Dataset

I didn’t use any predefined dataset from Kaggle or local storage —
I just went to Roboflow Universe
, searched for "mask detection" datasets, and picked a random one called
Face-mask-vsxay under the workspace “detection-and-segmentation”.

It was already labeled for two classes:

with_mask

without_mask

I used Roboflow’s API to download it directly in YOLOv8 format, which saves a lot of manual label conversion headaches.
The short script I used looked like this:

from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("detection-and-segmentation").project("face-mask-vsxay")
dataset = project.version(2).download("yolov8")
print("Dataset downloaded successfully!")


This automatically created a folder named something like Face-mask-2 that contained images/ and labels/ directories in YOLO format.

🧹 Step 2 — Preprocessing the Dataset

Once I had the dataset, I wanted to make it clean and consistent before training YOLO.
So I wrote a preprocess.py script that does a few important things:

Resize all images to 640×640 pixels (the standard YOLOv8 input size).

Split the dataset into:

Train (80%)

Validation (15%)

Test (5%)

Augment the training data by flipping some images horizontally — this helps YOLO generalize better.

Copy all labels properly so that each image still points to its .txt label.

Generate a data.yaml file that YOLOv8 needs to know where your train/val/test folders are.

Basically, after running:

python preprocess.py


it created a clean folder called dataset_processed/ with this structure:

dataset_processed/
├── images/train
├── images/val
├── images/test
├── labels/train
├── labels/val
├── labels/test
└── data.yaml


And the data.yaml file looked like this:

train: images/train
val: images/val
test: images/test
names:
  0: with_mask
  1: without_mask


This means YOLO now knows where to find everything and what each class ID represents.

⚙️ Step 3 — Training the Detection Model (YOLOv8)

Once my dataset was clean and ready, I jumped into training YOLOv8.
I used the YOLOv8n (nano) model since it’s fast and light — perfect for CPU or small GPU setups.

My train_yolo.py looked like this:

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # pretrained weights
model.train(
    data="dataset_processed/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    project="face_mask_yolo",
    name="run1"
)


That’s it! YOLOv8 takes care of the rest.
It logs training progress, saves the best weights, and gives you graphs like loss curves and mAP scores.

After training, the outputs are automatically saved under:

runs/detect/run1/
├── weights/
│   ├── best.pt
│   └── last.pt
└── results.png


The best.pt model is what I later used for evaluation and deployment.

🧠 Step 4 — Training the Classification Model (EfficientNet_B0)

Now, although YOLOv8 already detects mask vs no-mask quite well,
I wanted to add a second stage classifier to make the predictions even more confident.

So I used EfficientNet_B0 (from torchvision.models) — a lightweight CNN that’s great for binary classification tasks.

I trained it separately on cropped face images from the YOLO dataset.
Basically:

Input: single face image

Output: “With Mask” or “Without Mask”

Training setup:

from torchvision import models
import torch.nn as nn

model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)


Then I trained it for about 10–20 epochs using Adam optimizer and saved the best model as:

models/efficientnet_b0.pth


This classifier is later loaded inside the Flask app to classify cropped YOLO detections more precisely.

📊 Step 5 — Model Evaluation

For evaluation, I wanted to see how my model actually performs visually, not just by numbers.
So I wrote evaluate.py, which does two main things:

Draws YOLO predictions (bounding boxes + labels + confidence) on validation images and saves them in evaluation_outputs/.

Generates a Confusion Matrix using scikit-learn to see how many predictions were right/wrong.

Example run:

python evaluate.py


It automatically finds the best YOLO weights from the training folder and runs inference on all validation images.

Output:

evaluation_outputs/
├── image1_pred.jpg
├── image2_pred.jpg
└── confusion_matrix.png


The confusion matrix gives a clear picture of how well the model distinguishes between with_mask and without_mask.

🌐 Step 6 — Flask Web App Deployment

Once everything worked, I built a simple Flask web app (app.py) so anyone can test it easily.

Here’s what happens behind the scenes:

You upload an image on the Flask web UI.

The app loads YOLOv8 (for detection) and EfficientNet_B0 (for classification).

YOLO finds all the faces in the image.

Each detected face is cropped and sent to the EfficientNet model for classification.

The output image (with bounding boxes and mask labels) is displayed right on the page.

To run it:

python app.py


Then visit:

http://127.0.0.1:5000/


You’ll see a simple upload box → upload an image → get results instantly!

📁 Final Folder Overview

Here’s how my project looks after everything is set up:

face-mask-detection/
├── app.py                     # Flask app for inference
├── download_roboflow.py       # Dataset download
├── preprocess.py              # Resize, augment, and split dataset
├── train_yolo.py              # YOLOv8 training script
├── train_classifier.py        # EfficientNet_B0 training script
├── evaluate.py                # Confusion matrix + annotated results
│
├── dataset_processed/         # Clean, ready-to-train dataset
├── models/                    # Trained model weights
│   ├── yolov8_best.pt
│   └── efficientnet_b0.pth
├── runs/detect/run1/          # YOLOv8 training logs and weights
├── evaluation_outputs/        # Confusion matrix + visual results
└── static/uploads/            # Uploaded test images (via Flask)

✅ Summary — What I Did from Start to End

Picked a random Roboflow dataset (Face-mask-vsxay)

Downloaded it in YOLOv8 format using Roboflow API

Preprocessed the data — resized, augmented, and split

Trained YOLOv8 for mask detection

Trained EfficientNet_B0 for fine-grained mask classification

Evaluated the models — confusion matrix + annotated images

Deployed the models together in a Flask app

Tested locally by uploading random face images 🎉
