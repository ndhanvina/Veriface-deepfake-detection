import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ExifTags
from src.model import build_model
from src.utils import load_config, safe_load_json, safe_save_json
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
CONFIG = load_config()
CHECKPOINT_PATH = CONFIG["paths"]["checkpoint"]
FEEDBACK_FILE = CONFIG["paths"]["feedback"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["real", "fake"]

# ----------------------------
# Feedback file load/save (ALWAYS reload)
# ----------------------------
# Using safe_load_json and safe_save_json from utils for concurrency safety

# --- IMPORTANT ---
# Reload feedback memory on each rerun
feedback_memory = safe_load_json(FEEDBACK_FILE)


# ----------------------------
# IMAGE NORMALIZATION + HASH
# ----------------------------
def normalize_image_for_hash(img: Image.Image, size=(256, 256)):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = img._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation)
            if orientation_value == 3:
                img = img.rotate(180, expand=True)
            elif orientation_value == 6:
                img = img.rotate(270, expand=True)
            elif orientation_value == 8:
                img = img.rotate(90, expand=True)
    except:
        pass

    img = img.convert("RGB")
    img = img.resize(size, Image.LANCZOS)

    bio = BytesIO()
    img.save(bio, format="PNG")
    raw = bio.getvalue()

    return img, raw


def get_image_hash(image):
    norm_img, raw = normalize_image_for_hash(image)
    p_hash = imagehash.phash(norm_img)
    md5 = hashlib.md5(raw).hexdigest()[:10]
    return f"{str(p_hash)}-{md5}"


# ----------------------------
# FORENSIC CHECKS
# ----------------------------
def frequency_artifact_score(image):
    gray = np.array(image.convert("L"))
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    return float(np.mean(magnitude > np.percentile(magnitude, 99.5)))


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
    model = build_model(backbone=CONFIG["model"]["backbone"], num_classes=CONFIG["model"]["num_classes"])
    
    if os.path.exists(CHECKPOINT_PATH):
        # SECURITY FIX: weights_only=True
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)

        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt

        clean_state = {k.replace("model.", "").replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(clean_state, strict=False)
    else:
        st.warning(f"Checkpoint not found at {CHECKPOINT_PATH}. Using random weights.")
        
    model.to(DEVICE)
    model.eval()
    return model


# ----------------------------
# PREPROCESS
# ----------------------------
def preprocess_image(image):
    trans = transforms.Compose([
        transforms.Resize((CONFIG["data"]["img_size"], CONFIG["data"]["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return trans(image).unsqueeze(0)


# ----------------------------
# PREDICTION
# ----------------------------
def predict_image(model, image):
    img_hash = get_image_hash(image)

    # ---- apply memory instantly ----
    if img_hash in feedback_memory:
        return feedback_memory[img_hash]["label"], 1.0, img_hash

    tensor = preprocess_image(image).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
        prob = F.softmax(out, dim=1)
        conf, idx = torch.max(prob, 1)

    pred = CLASSES[idx.item()]
    conf = float(conf.item())

    freq = frequency_artifact_score(image)
    sharp = sharpness_score(image)
    noise = noise_level(image)

    heuristics = CONFIG["heuristics"]
    fake_score = (freq > heuristics["frequency_threshold"]) + \
                 (sharp < heuristics["sharpness_threshold"]) + \
                 (noise < heuristics["noise_threshold"])

    if fake_score >= heuristics["fake_score_threshold"]:
        return "fake", max(conf, 0.9), img_hash

    if pred == "real" and conf < heuristics["confidence_threshold_real"]:
        return "fake", 0.75, img_hash

    return pred, conf, img_hash


# ----------------------------
# SAVE FEEDBACK
# ----------------------------
def save_feedback(img_hash, correct_label):
    feedback_memory[img_hash] = {"label": correct_label}
    safe_save_json(FEEDBACK_FILE, feedback_memory)


# ----------------------------
# UI
# ----------------------------
st.title("DeepFake Detector (with Learning)")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_container_width=True)

    model = load_model()
    pred, conf, img_hash = predict_image(model, image)

    # ----------------------------
    # COLORFUL PREDICTIONS
    # ----------------------------
    st.subheader("Prediction")

    if pred == "real":
        st.success(f"REAL 🧍‍♂️ — Confidence: {conf*100:.2f}%")
    else:
        st.error(f"FAKE 🤖 — Confidence: {conf*100:.2f}%")

    # ----------------------------
    # FEEDBACK FORM
    # ----------------------------
    st.subheader("Is the prediction correct?")
    with st.form("feedback_form"):
        correct = st.radio("Correct label:", ["real", "fake"])
        submit = st.form_submit_button("Save Correction")

        if submit:
            save_feedback(img_hash, correct)
            st.success("Saved! The system will remember this forever.")
