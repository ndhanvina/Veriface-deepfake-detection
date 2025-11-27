import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ExifTags
from src.model import build_model
import json
import os
import imagehash
import numpy as np
import cv2
import hashlib
from io import BytesIO

# ----------------------------
# CONFIGURATION
# ----------------------------
CHECKPOINT_PATH = "outputs/best.ckpt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["real", "fake"]    # CORRECT ORDER
FEEDBACK_FILE = "feedback_memory.json"

# ----------------------------
# Helper: load/save feedback file
# ----------------------------
def load_feedback_file():
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w") as f:
            json.dump({}, f)
        return {}
    with open(FEEDBACK_FILE, "r") as f:
        try:
            return json.load(f)
        except Exception:
            return {}

def write_feedback_file(mem):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(mem, f, indent=4)

# initialize session feedback memory once
if "FEEDBACK_MEMORY" not in st.session_state:
    st.session_state.FEEDBACK_MEMORY = load_feedback_file()

# ----------------------------
# IMAGE NORMALIZATION + HASHING
# ----------------------------
def normalize_image_for_hash(img: Image.Image, size=(256, 256)):
    """
    Normalize image to remove EXIF orientation, force RGB and a fixed size.
    Returns a PIL image and its raw bytes.
    """
    # remove EXIF orientation if present
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = img._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation, None)
            if orientation_value == 3:
                img = img.rotate(180, expand=True)
            elif orientation_value == 6:
                img = img.rotate(270, expand=True)
            elif orientation_value == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass

    img = img.convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    # raw bytes for md5:
    bio = BytesIO()
    img.save(bio, format="PNG")
    raw = bio.getvalue()
    return img, raw

def get_image_hash(image: Image.Image):
    """
    Combined perceptual + MD5 hash for stability across uploads.
    """
    norm_img, raw = normalize_image_for_hash(image, size=(256, 256))
    # perceptual hash
    p_hash = imagehash.phash(norm_img)  # more robust than average_hash
    # md5 of normalized pixels
    md5 = hashlib.md5(raw).hexdigest()[:12]
    return f"{str(p_hash)}-{md5}"

# ----------------------------
# FORENSIC CHECKS
# ----------------------------
def frequency_artifact_score(image):
    gray = np.array(image.convert("L"))
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    score = np.mean(magnitude > np.percentile(magnitude, 99.5))
    return float(score)

def sharpness_score(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())

def noise_level(image):
    im = np.array(image.convert("L"))
    return float(np.std(im - cv2.medianBlur(im, 5)))

# ----------------------------
# MODEL LOADING
# ----------------------------
@st.cache_resource
def load_model():
    model = build_model(backbone="resnet18", num_classes=len(CLASSES))

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Clean prefixes (Lightning, DDP, etc.)
    clean_state = {}
    for k, v in state_dict.items():
        newk = k.replace("model.", "").replace("module.", "").replace("net.", "")
        clean_state[newk] = v

    model.load_state_dict(clean_state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# ----------------------------
# IMAGE PREPROCESSING (for model)
# ----------------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ----------------------------
# PREDICTION FUNCTION (uses session feedback)
# ----------------------------
def predict_image(model, image):
    # compute stable hash
    img_hash = get_image_hash(image)

    # direct memory override
    if img_hash in st.session_state.FEEDBACK_MEMORY:
        corrected = st.session_state.FEEDBACK_MEMORY[img_hash]
        return corrected["label"], 1.0, img_hash

    # model prediction
    tensor = preprocess_image(image).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    pred_label = CLASSES[pred.item()]
    base_conf = float(conf.item())

    # forensic signals
    freq = frequency_artifact_score(image)
    sharp = sharpness_score(image)
    noise = noise_level(image)

    fake_score = 0
    if freq > 0.012: fake_score += 1
    if sharp < 120: fake_score += 1
    if noise < 3.0: fake_score += 1

    # Overrides
    if fake_score >= 2:
        return "fake", max(base_conf, 0.90), img_hash

    if pred_label == "real" and base_conf < 0.60:
        return "fake", 0.75, img_hash

    return pred_label, base_conf, img_hash

# ----------------------------
# SAVE FEEDBACK (updates session memory & file)
# ----------------------------
def save_feedback(img_hash, correct_label):
    # update session memory immediately
    st.session_state.FEEDBACK_MEMORY[img_hash] = {"label": correct_label}
    # write file to disk
    write_feedback_file(st.session_state.FEEDBACK_MEMORY)

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="DeepFake Detector", layout="centered")
st.title("🕵️‍♂️ DeepFake Detection App")
st.markdown(
    "Upload an image and this app will predict whether it's **Real** or **Fake**.\n"
    "If the model gets it wrong, your correction improves future predictions."
)

uploaded_file = st.file_uploader("📤 Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # read image from uploaded file (working copy)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        model = load_model()
        pred, conf, img_hash = predict_image(model, image)

    st.subheader("🔍 Prediction Result")
    if pred == "real":
        st.success(f"**Prediction:** REAL 🧍‍♂️\n**Confidence:** {conf*100:.2f}%")
    else:
        st.error(f"**Prediction:** FAKE 🤖\n**Confidence:** {conf*100:.2f}%")

    st.markdown("### ❓ Was this prediction correct?")

    # Use a form so correction + save happen atomically in one interaction
    with st.form(key=f"correction_form_{img_hash}"):
        correction = st.radio("Correct label:", ["real", "fake"], index=0)
        submitted = st.form_submit_button("Save Correction" if pred in CLASSES else "Save Correction")
        if submitted:
            save_feedback(img_hash, correction)
            st.success("✅ Correction saved! Future predictions will improve for similar images.")

else:
    st.info("Please upload an image to get a prediction.")

st.caption("Model: ResNet18 + Forensic Checks | Memory: Feedback Learning | Interface: Streamlit")
