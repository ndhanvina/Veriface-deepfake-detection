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
CLASSES = ["real", "fake"]
FEEDBACK_FILE = "feedback_memory.json"


# ----------------------------
# Feedback file load/save (ALWAYS reload)
# ----------------------------
def load_feedback_file():
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w") as f:
            json.dump({}, f)
    try:
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    except:
        return {}


def save_feedback_file(mem):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(mem, f, indent=4)


# --- IMPORTANT ---
# Reload feedback memory on each rerun
feedback_memory = load_feedback_file()


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
    model = build_model(backbone="resnet18", num_classes=2)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    if "model" in ckpt:
        state = ckpt["model"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    clean_state = {k.replace("model.", "").replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


# ----------------------------
# PREPROCESS
# ----------------------------
def preprocess_image(image):
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
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

    # model prediction
    tensor = preprocess_image(image).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
        prob = F.softmax(out, dim=1)
        conf, idx = torch.max(prob, 1)

    pred = CLASSES[idx.item()]
    conf = float(conf.item())

    # forensic calculations
    freq = frequency_artifact_score(image)
    sharp = sharpness_score(image)
    noise = noise_level(image)

    fake_score = (freq > 0.012) + (sharp < 120) + (noise < 3.0)

    if fake_score >= 2:
        return "fake", max(conf, 0.9), img_hash

    if pred == "real" and conf < 0.6:
        return "fake", 0.75, img_hash

    return pred, conf, img_hash


# ----------------------------
# SAVE FEEDBACK
# ----------------------------
def save_feedback(img_hash, correct_label):
    feedback_memory[img_hash] = {"label": correct_label}
    save_feedback_file(feedback_memory)


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

    st.subheader("Prediction")
    st.write(f"**Label:** {pred}")
    st.write(f"**Confidence:** {conf*100:.2f}%")

    st.subheader("Is the prediction correct?")
    with st.form("feedback_form"):
        correct = st.radio("Correct label:", ["real", "fake"])
        submit = st.form_submit_button("Save Correction")

        if submit:
            save_feedback(img_hash, correct)
            st.success("Saved! The system will remember this forever.")
