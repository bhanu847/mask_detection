

# app.py
from flask import Flask, request, render_template_string, redirect, url_for, send_from_directory
import os
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = "uploads"
WEIGHTS_PATH = os.path.join("weights", "classifier.pth")  
CLASS_NAMES = ["without_mask", "with_mask"]
INPUT_SIZE = 224

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "secret123"

def load_model():
    model = models.efficientnet_b0(pretrained=True)
    try:
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, len(CLASS_NAMES))
    except Exception:
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, len(CLASS_NAMES))

    if os.path.exists(WEIGHTS_PATH):
        try:
            state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            new_state = {}
            for k, v in state.items():
                new_key = k.replace("module.", "") if k.startswith("module.") else k
                new_state[new_key] = v
            model.load_state_dict(new_state, strict=False)
            print(f"[INFO] Loaded custom weights from {WEIGHTS_PATH}")
        except Exception as e:
            print(f"[WARN] Could not load custom weights: {e}")
            print("[INFO] Using ImageNet-pretrained EfficientNet-B0 (last layer adapted).")
    else:
        print("[INFO] No custom weights found — using ImageNet-pretrained model (last layer adapted).")

    model.to(DEVICE)
    model.eval()
    return model


model = load_model()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def annotate_and_save(img_path, label_text, dest_dir=None):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=28)
    except Exception:
        font = ImageFont.load_default()

    text = label_text
    padding = 8
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        w, h = draw.textsize(text, font=font)

    rect_x0, rect_y0 = 10, 10
    rect_x1, rect_y1 = rect_x0 + w + padding * 2, rect_y0 + h + padding * 2

    draw.rectangle([(rect_x0, rect_y0), (rect_x1, rect_y1)], fill=(255, 255, 255, 230))
    draw.text((rect_x0 + padding, rect_y0 + padding), text, fill=(0, 0, 0), font=font)

    p = Path(img_path)
    annotated_name = f"{p.stem}_annotated{p.suffix}"
    if dest_dir is None:
        dest_dir = p.parent
    dest_path = Path(dest_dir) / annotated_name
    img.save(dest_path)
    return str(dest_path)


def predict_image(img_path, debug=True):
    """
    Returns: (label, conf, probs_list, annotated_path)
    probs_list is list of (class_name, prob)
    """
    img = Image.open(img_path).convert("RGB")
    inp = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(inp)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    probs_list = list(zip(CLASS_NAMES, probs.tolist()))
    idx = int(probs.argmax())
    label = CLASS_NAMES[idx]
    conf = float(probs[idx])

    if debug:
        print("[DEBUG] Prediction debug info for:", img_path)
        for name, p in probs_list:
            print(f"    {name}: {p:.6f}")
        print(f"    -> Predicted: {label} ({conf:.6f})")

    # annotate and save preview image inside uploads directory
    annotated_path = annotate_and_save(img_path, f"{label} ({conf*100:.1f}%)", dest_dir=app.config["UPLOAD_FOLDER"])
    annotated_filename = Path(annotated_path).name
    return label, conf, probs_list, annotated_filename

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Mask Classifier</title>
  <style>
    body {font-family: Arial, sans-serif; text-align:center; padding:40px; background:#f7f7f7;}
    .card {display:inline-block; background:white; padding:30px; border-radius:12px; box-shadow:0 6px 18px rgba(0,0,0,0.08);}
    h1 {color:#333;}
    input[type=file], input[type=submit] {
        margin-top:16px; padding:10px 14px; font-size:15px; border-radius:6px;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Face Mask Classifier</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" required><br>
      <input type="submit" value="Classify Image">
    </form>
  </div>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Result</title>
  <style>
    body {font-family: Arial, sans-serif; text-align:center; padding:40px; background:#f7f7f7;}
    .card {display:inline-block; background:white; padding:24px; border-radius:12px; box-shadow:0 6px 18px rgba(0,0,0,0.08); max-width:720px;}
    img {max-width:100%; height:auto; margin-top:8px; border-radius:10px; display:block; margin-left:auto; margin-right:auto;}
    .result {font-size:20px; margin-top:16px;}
    .label {font-weight:700;}
    ul {list-style:none; padding:0; margin-top:12px; text-align:left; display:inline-block;}
    li {padding:4px 0;}
    .back {display:block; margin-top:18px; color:#555; text-decoration:none;}
    .debug {font-size:12px; color:#666; margin-top:6px;}
  </style>
</head>
<body>
  <div class="card">
    <h2>Classification Result</h2>

    <!-- Show annotated image -->
    <img src="{{ url_for('uploaded_file', filename=filename) }}" alt="Uploaded Image">

    <!-- RESULT BELOW IMAGE -->
    <div class="result">
      <p><span class="label">Prediction:</span> {{ label }} &nbsp; (<strong>{{ "%.2f"|format(conf*100) }}%</strong>)</p>
    </div>

    <h4>Class Probabilities</h4>
    <ul>
      {% for name, p in probs %}
        <li>{{ name }}: {{ "%.3f"|format(p) }}</li>
      {% endfor %}
    </ul>

    <a class="back" href="{{ url_for('index') }}">&#8592; Upload another image</a>
    <div class="debug">
      <p>Tip: check server console for full debug printout. If classes look inverted, swap the order of <code>CLASS_NAMES</code> in the code to match your training label order.</p>
    </div>
  </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(url_for("index"))
        file = request.files["file"]
        if file.filename == "":
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"{Path(filename).stem}_{ts}{Path(filename).suffix}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], save_name)
        file.save(save_path)

        label, conf, probs, annotated_filename = predict_image(save_path)
        return render_template_string(RESULT_HTML, filename=annotated_filename, label=label, conf=conf, probs=probs)
    return render_template_string(INDEX_HTML)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


