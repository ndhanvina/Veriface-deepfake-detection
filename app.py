import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.model import build_model
import json
import os
import imagehash
import numpy as np
import cv2

# ----------------------------
# CONFIGURATION
# ----------------------------
CHECKPOINT_PATH = "outputs/best.ckpt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["real", "fake"]
FEEDBACK_FILE = "feedback_memory.json"

# ----------------------------
# LOAD / INIT FEEDBACK MEMORY
# ----------------------------
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump({}, f)

with open(FEEDBACK_FILE, "r") as f:
    FEEDBACK_MEMORY = json.load(f)

# ----------------------------
# FORENSIC CHECKS (NEW)
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
    return cv2.Laplacian(img, cv2.CV_64F).var()

def noise_level(image):
    im = np.array(image.convert("L"))
    return np.std(im - cv2.medianBlur(im, 5))

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

    clean_state = {k.replace("model.", "").replace("net.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(clean_state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# ----------------------------
# IMAGE PREPROCESSING
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
# IMAGE HASH
# ----------------------------
def get_image_hash(image):
    return str(imagehash.average_hash(image))

# ----------------------------
# UPDATED PREDICTION FUNCTION
# ----------------------------
def predict_image(model, image):
    img_hash = get_image_hash(image)

    # Memory correction
    if img_hash in FEEDBACK_MEMORY:
        corrected = FEEDBACK_MEMORY[img_hash]
        return corrected["label"], 1.0, img_hash

    # Model prediction
    tensor = preprocess_image(image).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    pred_label = CLASSES[pred.item()]
    base_conf = conf.item()

    # ----------------------------
    # FORENSIC ENSEMBLE LOGIC (NEW)
    # ----------------------------
    freq = frequency_artifact_score(image)
    sharp = sharpness_score(image)
    noise = noise_level(image)

    fake_score = 0
    if freq > 0.012: fake_score += 1
    if sharp < 120: fake_score += 1
    if noise < 3.0: fake_score += 1

    # If 2 out of 3 forensic tests say FAKE → override
    if fake_score >= 2:
        return "fake", max(base_conf, 0.90), img_hash

    # Real-but-low-confidence → likely fake
    if pred_label == "real" and base_conf < 0.60:
        return "fake", 0.75, img_hash

    return pred_label, base_conf, img_hash

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
    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Yes, correct"):
            st.success("Thank you! Model prediction confirmed.")

    with col2:
        if st.button("👎 No, wrong prediction"):
            st.warning("Please select the correct label below:")
            correction = st.radio("Correct label:", ["real", "fake"], horizontal=True)

            if st.button("Save Correction"):
                save_feedback(img_hash, correction)
                st.success("✅ Correction saved! Future predictions will improve.")

else:
    st.info("Please upload an image to get a prediction.")

st.caption("Model: ResNet18 + Forensic Checks | Memory: Feedback Learning | Interface: Streamlit")
