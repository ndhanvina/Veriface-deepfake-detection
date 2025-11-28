# predict.py (advanced, online-feedback capable)
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import argparse
import json
import os
from src.model import build_model
from src.utils import load_config, safe_load_json, safe_save_json
import warnings
import time

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Files & config ----------
CONFIG = load_config()
FEEDBACK_FILE = CONFIG["paths"]["feedback"]
MEMORY_MAX = 500             # max stored corrected examples
EMB_SIM_THRESHOLD = 0.75     # cosine similarity threshold for prototype match
PROTOTYPE_MOMENTUM = 0.85    # momentum when updating prototypes
ONLINE_TRAIN_STEPS = 12      # small number of SGD steps on feedback
ONLINE_BATCH_SIZE = 32
LR = 1e-3

# ---------- Embedding model (ResNet18; identity head) ----------
embed_model = models.resnet18(weights="IMAGENET1K_V1")
# replace fc with identity to get embedding vector
embed_dim = embed_model.fc.in_features
embed_model.fc = torch.nn.Identity()
embed_model.to(device)
embed_model.eval()

embed_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_embedding(img: Image.Image):
    t = embed_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = embed_model(t).cpu().numpy().flatten()
    norm = np.linalg.norm(emb) + 1e-10
    return emb / norm


# ---------- Main model transform ----------
main_transform = transforms.Compose([
    transforms.Resize((CONFIG["data"]["img_size"], CONFIG["data"]["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------- Utility: 5-crop generation ----------
def generate_crops(img, crop=224):
    w, h = img.size
    if w < crop or h < crop:
        img = img.resize((max(w, crop), max(h, crop)))
        w, h = img.size
    cx = (w - crop) // 2
    cy = (h - crop) // 2
    return [
        img.crop((0, 0, crop, crop)),
        img.crop((w - crop, 0, w, crop)),
        img.crop((0, h - crop, crop, h)),
        img.crop((w - crop, h - crop, w, h)),
        img.crop((cx, cy, cx + crop, cy + crop)),
    ]


# ---------- Load/save feedback memory ----------
def load_feedback():
    stored = safe_load_json(FEEDBACK_FILE, default={"entries": [], "prototypes": {}, "classifier": None})
    
    # convert lists back to numpy for runtime
    memory = {
        "entries": [
            {
                "embedding": np.array(e["embedding"], dtype=np.float32),
                "label": e["label"],
                "time": e.get("time", 0),
                "id": e.get("id", None)
            }
            for e in stored.get("entries", [])
        ],
        "prototypes": {k: np.array(v, dtype=np.float32) for k, v in stored.get("prototypes", {}).items()},
        "classifier": stored.get("classifier", None)  # will be dict of weights/bias
    }
    return memory


def save_feedback(memory):
    # convert numpy arrays to lists for JSON storing
    stored = {
        "entries": [
            {"embedding": e["embedding"].tolist(), "label": e["label"], "time": e["time"], "id": e.get("id")}
            for e in memory["entries"]
        ],
        "prototypes": {k: v.tolist() for k, v in memory.get("prototypes", {}).items()},
        "classifier": memory.get("classifier", None)
    }
    safe_save_json(FEEDBACK_FILE, stored)


# ---------- Online linear classifier in embedding space ----------
class OnlineLinearHead(torch.nn.Module):
    def __init__(self, emb_dim, num_classes=2):
        super().__init__()
        # small linear layer mapping embeddings -> logits
        self.fc = torch.nn.Linear(emb_dim, num_classes)
        # initialise small weights (helps stability)
        torch.nn.init.normal_(self.fc.weight, std=0.01)
        torch.nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        return self.fc(x)


def classifier_from_state(state_dict):
    # load from saved state (weights as lists)
    model = OnlineLinearHead(embed_dim, num_classes=2).to(device)
    if state_dict is None:
        return model
    w = np.array(state_dict["weight"], dtype=np.float32)
    b = np.array(state_dict["bias"], dtype=np.float32)
    model.fc.weight.data.copy_(torch.from_numpy(w))
    model.fc.bias.data.copy_(torch.from_numpy(b))
    return model


def classifier_to_state(model):
    return {
        "weight": model.fc.weight.detach().cpu().numpy().tolist(),
        "bias": model.fc.bias.detach().cpu().numpy().tolist()
    }


# ---------- Memory helpers: add / prune / prototype update ----------
def add_memory_example(memory, embedding, label, uid=None):
    # embedding: numpy array (normalized)
    entry = {"embedding": embedding.astype(np.float32), "label": label, "time": time.time(), "id": uid}
    memory["entries"].append(entry)
    # prune if needed (oldest first)
    if len(memory["entries"]) > MEMORY_MAX:
        memory["entries"] = memory["entries"][-MEMORY_MAX:]


def update_prototype(memory, embedding, label):
    # label: 'real' or 'fake'
    prototypes = memory.setdefault("prototypes", {})
    if label in prototypes:
        proto = prototypes[label]
        # momentum update (moving average)
        prototypes[label] = (PROTOTYPE_MOMENTUM * proto + (1 - PROTOTYPE_MOMENTUM) * embedding)
        # re-normalize
        prototypes[label] = prototypes[label] / (np.linalg.norm(prototypes[label]) + 1e-10)
    else:
        prototypes[label] = embedding.copy()


def knn_vote(memory, embedding, k=7):
    # returns (label, avg_similarity, counts) where counts is dict
    if not memory["entries"]:
        return None, 0.0, {}
    sims = []
    for e in memory["entries"]:
        sims.append((np.dot(embedding, e["embedding"]), e["label"]))
    sims.sort(key=lambda x: x[0], reverse=True)
    top = sims[:k]
    counts = {}
    total_sim = 0.0
    for s, lab in top:
        counts[lab] = counts.get(lab, 0) + 1
        total_sim += s
    if not top:
        return None, 0.0, {}
    avg_sim = total_sim / len(top)
    # pick majority by count (tie -> choose highest avg sim)
    best_label = max(counts.items(), key=lambda x: (x[1], x[0]))[0]
    return best_label, avg_sim, counts


# ---------- Apply feedback adjustment to model probs ----------
def apply_feedback_policy(probs, embedding, memory, classifier):
    """
    Combine base model probs with:
      - classifier head (trained online on embeddings)
      - prototype similarity bias
      - k-NN voting
    Returns adjusted probabilities.
    """
    # probs: numpy array [p_real, p_fake]
    adjusted = probs.copy()

    # 1) classifier logits on embedding
    if classifier is not None:
        classifier.eval()
        with torch.no_grad():
            x = torch.from_numpy(embedding).float().unsqueeze(0).to(device)
            logits = classifier(x).cpu().numpy().flatten()
            clf_probs = np.exp(logits) / np.sum(np.exp(logits))
        # combine with weight (classifier weight grows if we have memory)
        clf_w = 0.5 if memory["entries"] else 0.0
        adjusted = (1 - clf_w) * adjusted + clf_w * clf_probs

    # 2) prototype bias
    prototypes = memory.get("prototypes", {})
    if prototypes:
        for lab, proto in prototypes.items():
            sim = float(np.dot(embedding, proto))
            if sim > EMB_SIM_THRESHOLD:
                # boost the corresponding class depending on similarity
                idx = 0 if lab == "real" else 1
                adjusted[idx] += 0.15 * (sim - EMB_SIM_THRESHOLD)  # scaled boost

    # 3) k-NN
    knn_label, knn_sim, counts = knn_vote(memory, embedding, k=7)
    if knn_label is not None and knn_sim > EMB_SIM_THRESHOLD:
        idx = 0 if knn_label == "real" else 1
        adjusted[idx] += 0.12 * (knn_sim - EMB_SIM_THRESHOLD)

    # final normalization & smoothing
    adjusted = np.maximum(adjusted, 1e-8)
    adjusted = adjusted / adjusted.sum()
    return adjusted


# ---------- Load trained model checkpoint (your original loader) ----------
def load_model(checkpoint_path):
    model = build_model(backbone=CONFIG["model"]["backbone"], num_classes=CONFIG["model"]["num_classes"])
    
    if os.path.exists(checkpoint_path):
        # SECURITY FIX: weights_only=True
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
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
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights.")
        
    model.to(device)
    model.eval()
    return model


# ---------- Small online training function ----------
def online_finetune(classifier, memory, new_emb, new_label, steps=ONLINE_TRAIN_STEPS):
    """
    Train linear classifier a few steps using a small replay batch from memory + new example.
    classifier: OnlineLinearHead on device
    memory: memory dict with numpy embeddings
    new_emb: numpy embedding (normalized)
    new_label: 'real' or 'fake'
    """
    # build dataset: sample from memory uniformly up to batch size-1 and add new example
    examples = []
    labels = []
    # convert labels to 0/1
    def lab2idx(l): return 0 if l == "real" else 1
    # sample random examples
    mem_entries = memory["entries"]
    # shuffle copy to avoid altering original
    if mem_entries:
        idxs = np.random.choice(len(mem_entries), size=min(len(mem_entries), ONLINE_BATCH_SIZE - 1), replace=False)
        for i in idxs:
            examples.append(mem_entries[i]["embedding"])
            labels.append(lab2idx(mem_entries[i]["label"]))
    # add new example
    examples.append(new_emb)
    labels.append(lab2idx(new_label))
    X = torch.from_numpy(np.vstack(examples)).float().to(device)  # (N, D)
    y = torch.from_numpy(np.array(labels, dtype=np.int64)).to(device)

    optimizer = torch.optim.SGD(classifier.parameters(), lr=LR, momentum=0.9, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    classifier.train()
    for _ in range(steps):
        idx = np.random.randint(0, X.shape[0])
        xb = X[idx:idx + 1]
        yb = y[idx:idx + 1]
        logits = classifier(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # after training return classifier state dict (serializable)
    return classifier


# ---------- Main prediction pipeline ----------
def predict_image(model, image_path, memory, classifier):
    img = Image.open(image_path).convert("RGB")

    # model predictions via 5-crops ensemble
    crops = generate_crops(img)
    crop_probs = []
    for c in crops:
        t = main_transform(c).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t)
            p = torch.softmax(out, dim=1).cpu().numpy()[0]
            crop_probs.append(p)
    mean_probs = np.mean(crop_probs, axis=0)
    probs = mean_probs ** 1.25
    probs = probs / probs.sum()

    # get normalized embedding for similarity/classifier
    emb = get_embedding(img)  # numpy normalized vector

    # apply feedback policy using memory + classifier
    final_probs = apply_feedback_policy(probs, emb, memory, classifier)

    return final_probs, emb


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=CONFIG["paths"]["checkpoint"])
    args = parser.parse_args()

    # load models+memory
    model = load_model(args.checkpoint)
    memory = load_feedback()
    classifier_state = memory.get("classifier", None)
    classifier = classifier_from_state(classifier_state)

    probs, embedding = predict_image(model, args.image, memory, classifier)

    labels = ["REAL", "FAKE"]
    pred = int(np.argmax(probs))

    print("\n🧐 Prediction:", labels[pred])
    print(f"➡ REAL={probs[0]:.4f} | FAKE={probs[1]:.4f}")

    # Get feedback from user
    fb = input("\n❓ Was this correct? (yes/no): ").strip().lower()
    if fb == "no":
        correct = input("👉 Enter correct label (REAL/FAKE): ").strip().upper()
        if correct not in ("REAL", "FAKE"):
            print("Invalid label. aborting feedback.")
        else:
            correct_label = "real" if correct == "REAL" else "fake"
            # update memory
            add_memory_example(memory, embedding, correct_label, uid=str(hash(args.image)))
            update_prototype(memory, embedding, correct_label)

            # train online classifier with replay + new sample
            classifier = online_finetune(classifier, memory, embedding, correct_label, steps=ONLINE_TRAIN_STEPS)

            # save classifier state into memory for persistence
            memory["classifier"] = classifier_to_state(classifier)
            save_feedback(memory)
            print("\n✅ Saved correction and updated model memory (prototype + online head).")
    else:
        # Optionally: autopopulate memory for confident correct predictions to strengthen prototypes
        auto_confidence_threshold = 0.95
        if np.max(probs) > auto_confidence_threshold:
            # add pseudo-labeled example (optional)
            chosen = "real" if np.argmax(probs) == 0 else "fake"
            add_memory_example(memory, embedding, chosen, uid=str(hash(args.image)))
            update_prototype(memory, embedding, chosen)
            memory["classifier"] = classifier_to_state(classifier)
            save_feedback(memory)
            print("👍 Confident — stored pseudo-example to memory to help future similar images.")
        else:
            print("👍 Great!")

