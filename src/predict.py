import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import argparse
import json
import os
from src.model import build_model
import warnings

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------
# Feedback memory file
# --------------------------------------------
FEEDBACK_FILE = "feedback_memory.json"


# --------------------------------------------
# Load or create the feedback memory
# --------------------------------------------
def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return {}


def save_feedback(memory):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(memory, f, indent=4)


# --------------------------------------------
# Load trained model (.ckpt)
# --------------------------------------------
def load_model(checkpoint_path):
    model = build_model(backbone="resnet18", num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    clean_state = {k.replace("model.", "").replace("net.", ""): v
                    for k, v in state_dict.items()}

    model.load_state_dict(clean_state, strict=False)
    model.to(device)
    model.eval()
    return model


# --------------------------------------------
# Embedding model for feedback similarity
# --------------------------------------------
embed_model = models.resnet18(weights="IMAGENET1K_V1")
embed_model.fc = torch.nn.Identity()
embed_model.to(device)
embed_model.eval()

embed_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_embedding(img):
    t = embed_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = embed_model(t).cpu().numpy().flatten()

    # Normalize for cosine similarity
    norm = np.linalg.norm(emb)
    return emb / (norm + 1e-10)


# --------------------------------------------
# Main inference transform
# --------------------------------------------
main_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# --------------------------------------------
# Generate 5-crop inference (corner crops + center)
# --------------------------------------------
def generate_crops(img, crop=224):
    w, h = img.size

    if w < crop or h < crop:
        img = img.resize((max(w, crop), max(h, crop)))

    cx = (w - crop) // 2
    cy = (h - crop) // 2

    return [
        img.crop((0, 0, crop, crop)),
        img.crop((w - crop, 0, w, crop)),
        img.crop((0, h - crop, crop, h)),
        img.crop((w - crop, h - crop, w, h)),
        img.crop((cx, cy, cx + crop, cy + crop)),
    ]


# --------------------------------------------
# Apply feedback using embedding similarity
# --------------------------------------------
def apply_feedback(base_probs, embedding, memory):
    for img_hash, entry in memory.items():
        fb_emb = np.array(entry["embedding"])
        similarity = np.dot(embedding, fb_emb)

        if similarity > 0.75:  # similar to image seen earlier
            if entry["label"] == "real":
                base_probs[0] += 0.20
            else:
                base_probs[1] += 0.20

    base_probs = np.maximum(base_probs, 1e-8)
    base_probs = base_probs / base_probs.sum()
    return base_probs


# --------------------------------------------
# MAIN prediction function
# --------------------------------------------
def predict_image(model, image_path):
    img = Image.open(image_path).convert("RGB")

    crops = generate_crops(img)
    crop_probs = []

    for c in crops:
        t = main_transform(c).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t)
            p = torch.softmax(out, dim=1).cpu().numpy()[0]
            crop_probs.append(p)

    mean_probs = np.mean(crop_probs, axis=0)

    # Smooth probabilities
    probs = mean_probs ** 1.3
    probs = probs / probs.sum()

    # Get embedding for feedback
    emb = get_embedding(img)

    # Load memory and adjust
    memory = load_feedback()
    final_probs = apply_feedback(probs, emb, memory)

    return final_probs, emb


# --------------------------------------------
# CLI usage
# --------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="outputs/best.ckpt")
    args = parser.parse_args()

    model = load_model(args.checkpoint)

    probs, embedding = predict_image(model, args.image)

    labels = ["REAL", "FAKE"]
    pred = np.argmax(probs)

    print("\n🧐 Prediction:", labels[pred])
    print(f"➡ REAL={probs[0]:.4f} | FAKE={probs[1]:.4f}")

    # Feedback
    fb = input("\n❓ Was this correct? (yes/no): ").lower()

    if fb == "no":
        correct = input("👉 Enter correct label (REAL/FAKE): ").upper()
        memory = load_feedback()
        memory[str(hash(args.image))] = {
            "label": "real" if correct == "REAL" else "fake",
            "embedding": embedding.tolist()
        }
        save_feedback(memory)
        print("\n✅ Saved correction!")
    else:
        print("👍 Great!")
