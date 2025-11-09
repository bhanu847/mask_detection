from roboflow import Roboflow

rf = Roboflow(api_key="U3dxBFgpBuXshqMT3QR7")
project = rf.workspace("detection-and-segmentation").project("face-mask-vsxay")
version = project.version(2)
dataset = version.download("yolov8")
print("downloaded successfully....")
print("Dataset location:", dataset.location)








