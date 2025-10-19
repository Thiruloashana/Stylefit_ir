# main.py
import os
import pandas as pd
from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np

# Define directories
dataset_dir = "data"
images_dir = os.path.join(dataset_dir, "images", "pics")
csv_path = "aligned_dataset.csv"

# Load aligned DataFrame
df = pd.read_csv(csv_path)
# Convert captions back to list (saved as strings in CSV)
df['captions'] = df['captions'].apply(eval)  # Assumes captions were saved as stringified lists

# Initialize CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Process all data for embeddings (batch processing for efficiency)
def generate_embeddings(df, images_dir, model, processor, batch_size=32):
    all_image_embeds = []
    all_text_embeds = []
    
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        batch = df.iloc[start_idx:end_idx]
        image_inputs = []
        text_inputs = []

        for idx, row in batch.iterrows():
            img_path = os.path.join(images_dir, f"{row['candidate']}.png")
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                caption = row["captions"][0] if row["captions"] else "no caption"
                inputs = processor(text=caption, images=image, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
                image_inputs.append(inputs["pixel_values"])
                text_inputs.append(inputs["input_ids"])
            else:
                print(f"Image not found for candidate: {row['candidate']}")

        if image_inputs and text_inputs:
            inputs = {
                "pixel_values": torch.cat(image_inputs, dim=0),
                "input_ids": torch.cat(text_inputs, dim=0)
            }
            outputs = model(**inputs)
            all_image_embeds.append(outputs.image_embeds.detach().cpu().numpy())
            all_text_embeds.append(outputs.text_embeds.detach().cpu().numpy())
        else:
            print(f"No valid pairs in batch starting at index {start_idx}")

    return np.vstack(all_image_embeds), np.vstack(all_text_embeds)

# Build FAISS index
def build_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
    index.add(embeddings)
    return index

# Query the index
def query_index(index, query_embedding, k=5):
    distances, indices = index.search(query_embedding, k)
    return distances, indices

# Main execution
print(f"Loading data from {csv_path}")
image_embeds, text_embeds = generate_embeddings(df, images_dir, model, processor)

if image_embeds is not None and text_embeds is not None:
    print(f"Generated {image_embeds.shape[0]} image embeddings and {text_embeds.shape[0]} text embeddings.")
    
    # Build index for image embeddings
    image_index = build_index(image_embeds)
    print(f"FAISS index built with {image_index.ntotal} vectors.")

    # Example query (first image as query)
    query_embedding = image_embeds[0:1]  # Use first image as query
    distances, indices = query_index(image_index, query_embedding)
    print("Top 5 similar items (based on image):")
    for i, idx in enumerate(indices[0]):
        print(f"  Rank {i+1}: Candidate {df.iloc[idx]['candidate']} (Distance: {distances[0][i]:.4f})")

    print("Embedding and retrieval setup complete. Ready for StyleFit recommendations!")
else:
    print("Embedding generation failed. Check data or processing.")