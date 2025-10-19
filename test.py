#!/usr/bin/env python3
"""
test.py
=======
Runs retrieval for multiple candidate images/captions and visualizes top matches.

Author: StyleFit Team
"""

import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "models/clip_model"
INDEX_PATH = "data/target_index.faiss"
METADATA_PATH = "data/train_metadata.json"
DATA_PATH = "data/aligned_dataset.csv"
TEST_JSON = "try_test.json"  # your test file
TOP_K = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TEST_QUERIES = 10

print(DEVICE)

# -------------------------------
# LOAD MODEL + INDEX
# -------------------------------
print("Loading model and index...")
model = CLIPModel.from_pretrained(MODEL_PATH).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_PATH)
index = faiss.read_index(INDEX_PATH)

# -------------------------------
# LOAD METADATA
# -------------------------------
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r") as f:
        meta = json.load(f)
    TEXT_WEIGHT = meta.get("text_weight", 0.8)
    IMAGE_WEIGHT = meta.get("image_weight", 0.2)
else:
    TEXT_WEIGHT, IMAGE_WEIGHT = 0.8, 0.2

# -------------------------------
# LOAD DATASET (reference images)
# -------------------------------
df = pd.read_csv(DATA_PATH)
if "image_id" not in df.columns:
    df["image_id"] = df["image_path"].apply(lambda x: os.path.basename(x))
print(f"Loaded {len(df)} reference items.")

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def get_query_features(image_path=None, caption=None):
    if caption is None and image_path is None:
        raise ValueError("Provide at least caption or image_path.")

    image = None
    if image_path:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_path}: {e}")

    inputs = processor(
        text=[caption or ""],
        images=image if image is not None else None,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        txt_feat = model.get_text_features(inputs["input_ids"])
        txt_feat = F.normalize(txt_feat, p=2, dim=-1)
        if image is not None:
            img_feat = model.get_image_features(inputs["pixel_values"])
            img_feat = F.normalize(img_feat, p=2, dim=-1)
        else:
            img_feat = torch.zeros_like(txt_feat)

        combined = F.normalize(TEXT_WEIGHT * txt_feat + IMAGE_WEIGHT * img_feat, p=2, dim=-1)
    return combined.squeeze().cpu().numpy()

# -------------------------------
# RETRIEVAL FUNCTION
# -------------------------------
def retrieve_similar_items(query_emb, top_k=TOP_K):
    faiss.normalize_L2(query_emb.reshape(1, -1))
    distances, indices = index.search(query_emb.reshape(1, -1), top_k)
    return distances[0], indices[0]

# -------------------------------
# VISUALIZATION
# -------------------------------
def show_results(candidate_id, candidate_path, caption, target_id, target_path, score):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Candidate
    try:
        axes[0].imshow(Image.open(candidate_path))
    except:
        axes[0].text(0.5, 0.5, "Candidate image\nnot found", ha="center")
    axes[0].axis("off")
    axes[0].set_title(f"Candidate\nID: {candidate_id}\nCaption: {caption[:50]}")

    # Retrieved target
    try:
        axes[1].imshow(Image.open(target_path))
    except:
        axes[1].text(0.5, 0.5, "Target image\nnot found", ha="center")
    axes[1].axis("off")
    axes[1].set_title(f"Retrieved\nID: {target_id}\nScore: {score:.2f}")

    plt.tight_layout()
    plt.show()

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    print("Ready for batch testing...")

    # Load test JSON
    with open(TEST_JSON, "r") as f:
        test_entries = json.load(f)[:MAX_TEST_QUERIES]

    for entry in test_entries:
        candidate_id = entry["candidate"]
        captions = entry.get("captions", [])
        caption_text = " ".join(captions)

        # Candidate image path
        candidate_path = None
        for ext in ["png","jpg","jpeg","PNG","JPG","JPEG"]:
            path = os.path.join("data/images/pics", f"{candidate_id}.{ext}")
            if os.path.exists(path):
                candidate_path = path
                break
        if not candidate_path:
            print(f"Candidate image not found: {candidate_id}")
            continue

        # Generate query embedding
        query_emb = get_query_features(image_path=candidate_path, caption=caption_text)

        # Retrieve closest match
        distances, indices = retrieve_similar_items(query_emb, TOP_K)
        target_path = df.iloc[indices[0]]["image_path"]
        target_id = df.iloc[indices[0]]["image_id"]
        score = distances[0]

        # Print IDs and score
        print(f"Candidate ID: {candidate_id} --> Retrieved Target ID: {target_id}, Score: {score:.4f}")

        # Show result
        show_results(candidate_id, candidate_path, caption_text, target_id, target_path, score)
