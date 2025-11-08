'''from flask import Flask, request, render_template_string, redirect, url_for
import os
from pathlib import Path
from datetime import datetime
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models


UPLOAD_FOLDER = "uploads"
WEIGHTS_PATH = os.path.join("weights", "classifier.pth")  #optional
CLASS_NAMES = ["with_mask", "without_mask"]
INPUT_SIZE = 224


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "secret123"


def load_model():
    
    model = models.efficientnet_b0(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, len(CLASS_NAMES))

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
            print("[INFO] Using ImageNet-pretrained EfficientNet-B0.")
    else:
        print("[INFO] No custom weights found — using ImageNet-pretrained model.")

    model.to(DEVICE)
    model.eval()
    return model


model = load_model()

transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    inp = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(inp)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        label = CLASS_NAMES[idx]
        conf = float(probs[idx])
    return label, conf, list(zip(CLASS_NAMES, probs))


# ---------- HTML ----------
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Mask Classifier</title>
  <style>
    body {font-family: Arial, sans-serif; text-align:center; padding:50px;}
    h1 {color:#333;}
    input[type=file], input[type=submit] {
        margin-top:20px; padding:10px; font-size:16px;
    }
  </style>
</head>
<body>
  <h1>Face Mask Classifier</h1>
  <form method="POST" enctype="multipart/form-data">
    <input type="file" name="file" accept="image/*" required><br>
    <input type="submit" value="Classify Image">
  </form>
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
    body {font-family: Arial, sans-serif; text-align:center; padding:40px;}
    img {max-width:400px; margin-top:20px; border-radius:10px;}
    .result {font-size:22px; margin-top:20px; color:#008000;}
    ul {list-style:none; padding:0;}
  </style>
</head>
<body>
  <h1>Classification Result</h1>
  <img src="{{ url_for('uploaded_file', filename=filename) }}" alt="Uploaded Image">
  <div class="result">
    <p><b>Prediction:</b> {{ label }} ({{ "%.2f"|format(conf*100) }}%)</p>
  </div>
  <h3>Class Probabilities:</h3>
  <ul>
    {% for name, p in probs %}
      <li>{{ name }}: {{ "%.3f"|format(p) }}</li>
    {% endfor %}
  </ul>
  <a href="{{ url_for('index') }}">&#8592; Upload another image</a>
</body>
</html>
"""


# ---------- ROUTES ----------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file.filename == "":
            return redirect(url_for("index"))
        filename = Path(file.filename).name
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"{Path(filename).stem}_{ts}{Path(filename).suffix}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], save_name)
        file.save(save_path)

        label, conf, probs = predict_image(save_path)
        return render_template_string(RESULT_HTML, filename=save_name, label=label, conf=conf, probs=probs)
    return render_template_string(INDEX_HTML)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return redirect(url_for("static", filename=f"../uploads/{filename}"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)'''



# app.py
'''from flask import Flask, request, render_template_string, redirect, url_for, send_from_directory
import os
from pathlib import Path
from datetime import datetime
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from werkzeug.utils import secure_filename

# ---------- CONFIG ----------
UPLOAD_FOLDER = "uploads"
WEIGHTS_PATH = os.path.join("weights", "classifier.pth")  # optional
CLASS_NAMES = ["without_mask", "with_mask"]
INPUT_SIZE = 224

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "secret123"


# ---------- MODEL ----------
def load_model():
    model = models.efficientnet_b0(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, len(CLASS_NAMES))

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
            print("[INFO] Using ImageNet-pretrained EfficientNet-B0.")
    else:
        print("[INFO] No custom weights found — using ImageNet-pretrained model.")

    model.to(DEVICE)
    model.eval()
    return model


model = load_model()

transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    inp = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(inp)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        label = CLASS_NAMES[idx]
        conf = float(probs[idx])
    # return label, confidence, and list of (class, prob)
    return label, conf, list(zip(CLASS_NAMES, probs))


# ---------- HTML ----------
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
  </style>
</head>
<body>
  <div class="card">
    <h2>Classification Result</h2>

    <!-- IMAGE (shown first) -->
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
  </div>
</body>
</html>
"""


# ---------- ROUTES ----------
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

        label, conf, probs = predict_image(save_path)
        return render_template_string(RESULT_HTML, filename=save_name, label=label, conf=conf, probs=probs)
    return render_template_string(INDEX_HTML)


# Serve uploaded files from uploads/ directory
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    # debug=True only for development; remove or set False for production
    app.run(host="0.0.0.0", port=5000, debug=True)'''




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

# ---------- CONFIG ----------
UPLOAD_FOLDER = "uploads"
WEIGHTS_PATH = os.path.join("weights", "classifier.pth")  # optional
# NOTE: if predictions look inverted, swap this order to match training label order
CLASS_NAMES = ["without_mask", "with_mask"]
INPUT_SIZE = 224

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "secret123"


# ---------- MODEL ----------
def load_model():
    model = models.efficientnet_b0(pretrained=True)
    # adapt classifier to number of classes
    try:
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, len(CLASS_NAMES))
    except Exception:
        # fallback for slightly different torchvision versions
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

# Improved transform: resize shorter side to 256, center crop to 224
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# Annotate and save an image copy with label text
def annotate_and_save(img_path, label_text, dest_dir=None):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try to use a TTF font, fallback to default
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=28)
    except Exception:
        font = ImageFont.load_default()

    text = label_text
    padding = 8

    # ✅ Pillow ≥10.0 uses textbbox() instead of textsize()
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # fallback for older Pillow versions
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


# ---------- HTML ----------
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


# ---------- ROUTES ----------
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
        # annotated_filename is stored in uploads directory
        return render_template_string(RESULT_HTML, filename=annotated_filename, label=label, conf=conf, probs=probs)
    return render_template_string(INDEX_HTML)


# Serve uploaded files from uploads/ directory
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    # debug=True only for development; remove or set False for production
    app.run(host="0.0.0.0", port=5000, debug=True)





# app.py
'''from flask import Flask, request, render_template_string, redirect, url_for, send_from_directory
import os
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from werkzeug.utils import secure_filename
import json

# ---------- CONFIG ----------
UPLOAD_FOLDER = "uploads"
WEIGHTS_PATH = os.path.join("weights", "classifier.pth")  # optional
# Default order; if your training used a different mapping, you may need to swap.
CLASS_NAMES = ["without_mask", "with_mask"]
INPUT_SIZE = 224
MAPPING_CONFIG = "mapping.cfg"  # stores {"flipped": true/false}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "secret123"

# load persisted mapping preference if any
def load_mapping_pref():
    if os.path.exists(MAPPING_CONFIG):
        try:
            with open(MAPPING_CONFIG, "r") as f:
                cfg = json.load(f)
            return bool(cfg.get("flipped", False))
        except Exception:
            return False
    return False

def save_mapping_pref(flipped: bool):
    with open(MAPPING_CONFIG, "w") as f:
        json.dump({"flipped": bool(flipped)}, f)

MAPPING_FLIPPED = load_mapping_pref()
if MAPPING_FLIPPED:
    CLASS_NAMES = list(reversed(CLASS_NAMES))

# ---------- MODEL ----------
def load_model():
    model = models.efficientnet_b0(pretrained=True)
    # adapt classifier to number of classes (handles small torchvision API differences)
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
            print("[INFO] Using ImageNet-pretrained EfficientNet-B0 (classifier adapted).")
    else:
        print("[INFO] No custom weights found — using ImageNet-pretrained model (classifier adapted).")

    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Improved transform: resize shorter side to 256, center crop to 224
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Annotate and save an image copy with label text
def annotate_and_save(img_path, label_text, dest_dir=None):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=26)
    except Exception:
        font = ImageFont.load_default()

    text = label_text
    padding = 8
    # Pillow >=10: use textbbox; older: fallback to textsize
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

# Predict returns probs array (not mapped), so we can show both mappings
def predict_probs(img_path):
    img = Image.open(img_path).convert("RGB")
    inp = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(inp)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs  # e.g. [0.21, 0.79]

# ---------- HTML ----------
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
    .note {font-size:13px; color:#666; margin-top:10px;}
  </style>
</head>
<body>
  <div class="card">
    <h1>Face Mask Classifier</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" required><br>
      <input type="submit" value="Classify Image">
    </form>
    <div class="note">
      <p>If results look inverted, use the 'Use reversed mapping' button on the result page.</p>
    </div>
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
    body {font-family: Arial, sans-serif; text-align:center; padding:30px; background:#fff;}
    .card {display:inline-block; background:white; padding:24px; border-radius:12px; box-shadow:0 6px 18px rgba(0,0,0,0.08); max-width:820px;}
    img {max-width:100%; height:auto; margin-top:8px; border-radius:10px; display:block; margin-left:auto; margin-right:auto;}
    .result {font-size:20px; margin-top:16px;}
    .label {font-weight:700;}
    .two {display:flex; gap:20px; justify-content:center; margin-top:18px;}
    .col {flex:1; text-align:left; max-width:360px;}
    ul {list-style:none; padding:0; margin-top:12px;}
    li {padding:4px 0;}
    .back {display:block; margin-top:18px; color:#555; text-decoration:none;}
    .debug {font-size:13px; color:#666; margin-top:8px;}
    .btn {display:inline-block; padding:8px 12px; border-radius:6px; background:#007bff; color:white; text-decoration:none;}
  </style>
</head>
<body>
  <div class="card">
    <h1>Classification Result</h1>

    <!-- annotated image -->
    <img src="{{ url_for('uploaded_file', filename=annotated_filename) }}" alt="Annotated Image">

    <div class="result" style="text-align:center;">
      <p><span class="label">Prediction (current mapping):</span> {{ pred_name }} &nbsp; (<strong>{{ "%.2f"|format(pred_conf*100) }}%</strong>)</p>
    </div>

    

    <div style="text-align:center; margin-top:14px;">
      <!-- button to permanently flip mapping -->
      {% if mapping_flipped %}
        <a class="btn" href="{{ url_for('', action='off') }}">Use original mapping</a>
      {% else %}
        <a class="btn" href="{{ url_for('', action='on') }}">Use reversed mapping</a>
      {% endif %}
    </div>

    <a class="back" href="{{ url_for('index') }}">&#8592; Upload another image</a>

    <div class="debug">
      <p>   .</p>
      <p>Weights present: {{ weights_present }} &nbsp; | &nbsp; Mapping persisted: {{ mapping_flipped }}</p>
    </div>
  </div>
</body>
</html>
"""

# ---------- ROUTES ----------
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

        # compute probabilities (raw)
        probs = predict_probs(save_path)  # e.g. [0.52, 0.48]

        # mapped using current CLASS_NAMES
        mapped = list(zip(CLASS_NAMES, probs.tolist()))
        idx = int(probs.argmax())
        pred_name = CLASS_NAMES[idx]
        pred_conf = float(probs[idx])

        # reversed mapping: just reverse the class names but keep same probs order
        reversed_names = list(reversed(CLASS_NAMES))
        probs_reversed = list(zip(reversed_names, probs.tolist()))
        # best under reversed mapping would be:
        idx_rev = int(probs.argmax())
        pred_name_rev = reversed_names[idx_rev]
        pred_conf_rev = float(probs[idx_rev])

        # annotate with the primary prediction label (current mapping)
        annotated_path = annotate_and_save(save_path, f"{pred_name} ({pred_conf*100:.1f}%)", dest_dir=app.config["UPLOAD_FOLDER"])
        annotated_filename = Path(annotated_path).name

        weights_present = os.path.exists(WEIGHTS_PATH)
        return render_template_string(
            RESULT_HTML,
            annotated_filename=annotated_filename,
            pred_name=pred_name,
            pred_conf=pred_conf,
            probs_mapped=mapped,
            probs_reversed=probs_reversed,
            mapping_flipped=MAPPING_FLIPPED,
            weights_present=weights_present
        )

    return render_template_string(INDEX_HTML)


@app.route("/toggle_mapping/<action>")
def toggle_mapping(action="on"):
    global MAPPING_FLIPPED, CLASS_NAMES, model
    if action == "on":
        MAPPING_FLIPPED = True
    else:
        MAPPING_FLIPPED = False

    # update persisted config and update CLASS_NAMES in memory
    save_mapping_pref(MAPPING_FLIPPED)
    # reset class names to base order and then apply flip if needed
    base = ["without_mask", "with_mask"]
    CLASS_NAMES = list(reversed(base)) if MAPPING_FLIPPED else base

    # No need to reload model weights since only label mapping changed,
    # but if your classifier layer size depends on len(CLASS_NAMES) you would reload here.
    return redirect(url_for("index"))


# Serve uploaded files from uploads/ directory
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    print("[APP START] DEVICE:", DEVICE)
    print("[APP START] WEIGHTS_PATH exists:", os.path.exists(WEIGHTS_PATH))
    print("[APP START] Current CLASS_NAMES order:", CLASS_NAMES)
    app.run(host="0.0.0.0", port=5000, debug=True)

'''