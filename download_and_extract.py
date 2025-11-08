from roboflow import Roboflow

# Initialize Roboflow with your API key
rf = Roboflow(api_key="U3dxBFgpBuXshqMT3QR7")

# Connect to your workspace and project
project = rf.workspace("detection-and-segmentation").project("face-mask-vsxay")

# Specify which dataset version you want to download
version = project.version(2)

# Download dataset in YOLOv8 format
dataset = version.download("yolov8")

print("✅ Dataset downloaded successfully!")
print("📁 Dataset location:", dataset.location)




'''import zipfile, os

zip_path = "mask_dataset.zip"       # replace with your actual filename
extract_to = "mask_dataset_raw"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("✅ Extracted to:", os.path.abspath(extract_to))'''

