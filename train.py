#!/usr/bin/env python3
"""
train.py
========
Builds and saves a multimodal embedding model (CLIP-based) for image-text retrieval.

- Extracts CLIP embeddings for target images and captions
- Combines features with strong weight on captions
- Builds and saves FAISS index + model checkpoints

Author: StyleFit Team
"""

import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
import faiss
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# -------------------------------
# CONFIG
# -------------------------------
DATA_PATH = "data/aligned_dataset.csv"
INDEX_SAVE_PATH = "data/target_index.faiss"
MODEL_SAVE_PATH = "models/clip_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_WEIGHT = 0.8  # heavy weight for captions
IMAGE_WEIGHT = 0.2

# -------------------------------
# PREPARE DIRECTORIES
# -------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} samples.")

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def get_features(image_path, caption):
    """Compute weighted CLIP embeddings for image and caption."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

    inputs = processor(text=[caption], images=image, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        img_feat = model.get_image_features(inputs["pixel_values"])
        txt_feat = model.get_text_features(inputs["input_ids"])

    img_feat = F.normalize(img_feat, p=2, dim=-1)
    txt_feat = F.normalize(txt_feat, p=2, dim=-1)

    # Weighted fusion — heavy emphasis on text
    combined = F.normalize(TEXT_WEIGHT * txt_feat + IMAGE_WEIGHT * img_feat, p=2, dim=-1)
    return combined.squeeze().cpu().numpy()

# -------------------------------
# BUILD FAISS INDEX
# -------------------------------
features = []
valid_indices = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    emb = get_features(row["image_path"], row["captions"])
    if emb is not None:
        features.append(emb)
        valid_indices.append(i)

features = torch.tensor(features).numpy()
dim = features.shape[1]
print(f"Feature matrix shape: {features.shape}")

index = faiss.IndexFlatIP(dim)
faiss.normalize_L2(features)
index.add(features)
faiss.write_index(index, INDEX_SAVE_PATH)
print(f"Saved FAISS index to {INDEX_SAVE_PATH}")

# -------------------------------
# SAVE MODEL CHECKPOINT
# -------------------------------
model.save_pretrained(MODEL_SAVE_PATH)
processor.save_pretrained(MODEL_SAVE_PATH)
print(f"Saved CLIP model and processor to {MODEL_SAVE_PATH}")

# -------------------------------
# SAVE METADATA (OPTIONAL)
# -------------------------------
metadata = {
    "text_weight": TEXT_WEIGHT,
    "image_weight": IMAGE_WEIGHT,
    "num_vectors": len(features),
    "index_path": INDEX_SAVE_PATH,
    "model_path": MODEL_SAVE_PATH
}
with open("data/train_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("✅ Training complete. Model and index saved.")
