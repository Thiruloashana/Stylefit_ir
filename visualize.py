#!/usr/bin/env python3
"""
StyleFit Retrieval & Visualization (Candidate -> Predicted Target)
=================================================================

- Builds embeddings for target images listed in data/aligned_dataset.csv
- Reads test queries from JSON files ending with "test.json" inside captions dir
  (each entry is expected to include "candidate" and "captions")
- For each test entry: load candidate image from data/images/pics, encode candidate image + caption,
  search FAISS index of target images, return closest target id
- Visualize results: candidate id, candidate image, caption, predicted target id, target image
- Saves visualization per test file.

Usage:
    python retrieve_and_visualize.py --dataset_dir data --model_name openai/clip-vit-base-patch32
"""
import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import faiss
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self,
                 dataset_dir: str = "data",
                 model_name: str = "openai/clip-vit-base-patch32",
                 device: Optional[str] = None):
        self.dataset_dir = dataset_dir
        self.captions_dir = os.path.join(dataset_dir, "captions")
        self.images_dir = os.path.join(dataset_dir, "images", "pics")  # candidate images here
        self.index_images_dir = os.path.join(dataset_dir, "images", "pics")  # same dir used for index images by default
        self.csv_path = os.path.join(dataset_dir, "aligned_dataset.csv")  # contains target images to index
        self.embeds_path = os.path.join(dataset_dir, "target_embeds.npy")
        self.metadata_path = os.path.join(dataset_dir, "target_index_metadata.npy")  # store image ids ordering
        self.index_path = os.path.join(dataset_dir, "target_index.faiss")

        self.model_name = model_name
        self.device = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        logger.info(f"Using device: {self.device}")
        self._load_model()
        self.df = self._load_csv()

    def _load_model(self):
        try:
            logger.info(f"Loading CLIP model {self.model_name}")
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

    def _load_csv(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Aligned CSV not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        # normalize column names expectations (candidate, image_id/target, captions)
        # allow different column names by falling back
        if "target" in df.columns:
            df = df.rename(columns={"target": "image_id"})
        if "image_id" not in df.columns:
            raise ValueError("aligned_dataset.csv must contain 'image_id' or 'target' column")
        if "candidate" not in df.columns:
            raise ValueError("aligned_dataset.csv must contain 'candidate' column")
        if "captions" not in df.columns:
            logger.warning("No 'captions' column in CSV - continuing, but captions may be empty for some rows.")
            df["captions"] = [[] for _ in range(len(df))]

        # make sure captions are lists
        def _ensure_list(x):
            if pd.isna(x):
                return []
            if isinstance(x, str):
                x = x.strip()
                if x.startswith("[") and x.endswith("]"):
                    try:
                        parsed = eval(x)
                        return parsed if isinstance(parsed, list) else [x]
                    except Exception:
                        return [x]
                else:
                    return [x]
            return list(x) if isinstance(x, (list, tuple)) else [x]

        df["captions"] = df["captions"].apply(_ensure_list)
        # Add 'image_path' if missing: attempt to find image file in images_dir using image_id
        if "image_path" not in df.columns:
            df["image_path"] = df["image_id"].apply(lambda iid: self._find_image(iid))
        else:
            # ensure missing paths are attempted
            df["image_path"] = df.apply(lambda r: r["image_path"] if pd.notna(r["image_path"]) and os.path.exists(r["image_path"]) else self._find_image(r["image_id"]), axis=1)

        # Drop rows without valid image path
        before = len(df)
        df = df.dropna(subset=["image_path"]).reset_index(drop=True)
        logger.info(f"Loaded CSV with {before} rows, {len(df)} rows have valid image paths")
        return df

    def _find_image(self, image_id: str) -> Optional[str]:
        if pd.isna(image_id):
            return None
        image_id = str(image_id)
        for ext in ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]:
            p = os.path.join(self.index_images_dir, f"{image_id}.{ext}")
            if os.path.exists(p):
                return p
        return None

    def _load_image_safe(self, path: str) -> Optional[Image.Image]:
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")
            return None

    def build_or_load_index(self, recalc_embeddings: bool = False, use_cache: bool = True) -> faiss.Index:
        """
        Build FAISS index for all target images (from aligned_dataset.csv).
        Saves embeddings to npy and index file for reuse.
        """
        # If index exists and not recalcing, try to load
        if use_cache and os.path.exists(self.index_path) and os.path.exists(self.embeds_path) and os.path.exists(self.metadata_path) and not recalc_embeddings:
            try:
                logger.info("Loading existing FAISS index and embeddings from disk...")
                index = faiss.read_index(self.index_path)
                logger.info(f"Loaded FAISS index with {index.ntotal} vectors.")
                return index
            except Exception as e:
                logger.warning(f"Failed loading FAISS index: {e} — will rebuild.")

        logger.info("Encoding target images (from aligned CSV) to build index...")
        # Prepare images list and ids
        image_paths = self.df["image_path"].tolist()
        image_ids = self.df["image_id"].astype(str).tolist()

        all_feats = []
        batch = []
        batch_size = 32
        for i, p in enumerate(tqdm(image_paths, desc="Encoding targets")):
            img = self._load_image_safe(p)
            if img is None:
                all_feats.append(None)
                continue
            batch.append((i, img))
            if len(batch) >= batch_size:
                feats = self._encode_images([b[1] for b in batch])
                for idx, f in zip([b[0] for b in batch], feats):
                    all_feats.append((idx, f))
                batch = []
        if batch:
            feats = self._encode_images([b[1] for b in batch])
            for idx, f in zip([b[0] for b in batch], feats):
                all_feats.append((idx, f))

        # allocate array of valid embeddings in the same order as df rows (skip missing)
        embeddings_list = []
        meta_ids = []
        for i, p in enumerate(image_paths):
            # find feature corresponding to i in all_feats
            feat_entry = next((f for (idx, f) in all_feats if idx == i), None) if isinstance(all_feats[0], tuple) else None
            # But above logic is messy: better re-encode sequentially to guarantee order:
            pass

        # Simpler and safer: re-encode sequentially (ensures order)
        embeddings = []
        for p in tqdm(image_paths, desc="Encoding targets sequential"):
            img = self._load_image_safe(p)
            if img is None:
                embeddings.append(None)
                meta_ids.append(None)
                continue
            feat = self._encode_images([img])[0]
            embeddings.append(feat)
            meta_ids.append(Path(p).stem)

        # Filter missing
        filtered_embeddings = [e for e in embeddings if e is not None]
        filtered_ids = [i for e, i in zip(embeddings, meta_ids) if e is not None]

        if not filtered_embeddings:
            raise ValueError("No target embeddings could be generated - check your target images.")

        emb_matrix = np.vstack(filtered_embeddings).astype("float32")
        # normalize for cosine (use inner product on normalized vectors)
        emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)

        dim = emb_matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb_matrix)
        faiss.write_index(index, self.index_path)
        np.save(self.embeds_path, emb_matrix)
        np.save(self.metadata_path, np.array(filtered_ids, dtype=object))
        logger.info(f"Built index with {index.ntotal} vectors and saved to {self.index_path}")

        return index

    def _encode_images(self, pil_images: List[Image.Image]) -> np.ndarray:
        """
        Encode a list of PIL images to CLIP image features (CPU numpy array).
        """
        with torch.no_grad():
            inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            img_feats = self.model.get_image_features(**inputs)
            img_feats = F.normalize(img_feats, p=2, dim=1)
            return img_feats.cpu().numpy()

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            txt_feats = self.model.get_text_features(**inputs)
            txt_feats = F.normalize(txt_feats, p=2, dim=1)
            return txt_feats.cpu().numpy()

    def encode_candidate_query(self, candidate_image_path: str, caption: str) -> np.ndarray:
        """
        Create a combined embedding from candidate image and caption text.
        Returns a single normalized vector (numpy array shape (d,))
        """
        img = self._load_image_safe(candidate_image_path)
        if img is None:
            raise FileNotFoundError(f"Candidate image not found or invalid: {candidate_image_path}")
        img_feat = self._encode_images([img])[0]  # 1D
        txt_feat = self._encode_texts([caption])[0]
        combined = (img_feat + txt_feat) / 2.0
        combined = combined / np.linalg.norm(combined)
        return combined.astype("float32")

    def load_test_queries(self) -> Dict[str, List[Dict]]:
        """
        Read all JSON files in captions_dir that end with 'test.json'
        Returns mapping: filename -> list of {candidate, caption}
        expects each entry in test.json to have 'candidate' and 'captions' (or 'caption')
        """
        tests = {}
        if not os.path.exists(self.captions_dir):
            logger.warning(f"Captions dir not found: {self.captions_dir}")
            return tests

        files = [f for f in os.listdir(self.captions_dir) if f.endswith("test.json")]
        logger.info(f"Found {len(files)} test json files: {files}")
        for f in files:
            path = os.path.join(self.captions_dir, f)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as e:
                logger.warning(f"Failed to parse {path}: {e}")
                continue

            entries = []
            for entry in data:
                candidate = entry.get("candidate") or entry.get("candidateid") or entry.get("candidate_id")
                captions = entry.get("captions") or entry.get("caption") or []
                if isinstance(captions, list):
                    caption = " ".join(captions) if captions else ""
                else:
                    caption = str(captions)
                if candidate:
                    entries.append({"candidate": str(candidate), "caption": caption})
            tests[f] = entries
        return tests

    def search(self, query_vec: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Search built index (loads metadata) and return distances, indices, and corresponding target ids
        """
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            raise FileNotFoundError("Index or metadata not found - call build_or_load_index() first.")

        index = faiss.read_index(self.index_path)
        distances, indices = index.search(query_vec.reshape(1, -1), k)
        ids = np.load(self.metadata_path, allow_pickle=True)
        # indices are into filtered embeddings array; map to ids
        retrieved_ids = [ids[idx] if idx < len(ids) else None for idx in indices[0]]
        return distances, indices, retrieved_ids

    def visualize_results(self, results: List[Dict], out_path: str = "results.png", cols: int = 2):
        """
        results: list of dicts with keys:
          candidate_id, candidate_path, caption, predicted_target_id, predicted_target_path
        Visualize as grid of pairs (candidate | predicted target) with labels.
        """
        if not results:
            logger.info("No results to visualize.")
            return

        pairs = results
        rows = (len(pairs) + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols * 2, figsize=(6 * cols * 1.2, 4 * rows * 1.2))
        axs = axs.reshape(rows, cols * 2)

        for idx, res in enumerate(pairs):
            r = idx // cols
            c = (idx % cols) * 2
            # Candidate image
            ax_c = axs[r, c]
            try:
                img_c = self._load_image_safe(res["candidate_path"])
                ax_c.imshow(img_c)
            except Exception:
                ax_c.text(0.5, 0.5, "Candidate image\nnot found", ha="center")
            ax_c.axis("off")
            ax_c.set_title(f"Candidate: {res['candidate_id']}\nCaption: {res['caption'][:80]}", fontsize=8)

            # Predicted target image
            ax_t = axs[r, c + 1]
            try:
                img_t = self._load_image_safe(res["predicted_target_path"])
                ax_t.imshow(img_t)
            except Exception:
                ax_t.text(0.5, 0.5, "Target image\nnot found", ha="center")
            ax_t.axis("off")
            ax_t.set_title(f"Predicted target: {res['predicted_target_id']}", fontsize=8)

        # hide any unused axes
        total_axes = rows * cols * 2
        for extra in range(len(pairs), total_axes // 2):
            pass  # nothing needed; leftover axes will be processed below

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.show()
        logger.info(f"Saved visualization to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="StyleFit retrieval & visualization")
    parser.add_argument("--dataset_dir", default="data", help="Dataset directory")
    parser.add_argument("--model_name", default="openai/clip-vit-base-patch32", help="CLIP model")
    parser.add_argument("--k", type=int, default=1, help="Top-k retrieval")
    parser.add_argument("--recalc_index", action="store_true", help="Recalculate target index and embeddings")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test queries per test file for quick runs")
    args = parser.parse_args()

    retriever = Retriever(dataset_dir=args.dataset_dir, model_name=args.model_name)

    # Build or load index of target images (from aligned dataset)
    retriever.build_or_load_index(recalc_embeddings=args.recalc_index)

    # Load test queries from test.json files
    tests = retriever.load_test_queries()
    if not tests:
        logger.warning("No test files found. Exiting.")
        return

    # For each test file, perform retrieval and visualize
    for test_fname, entries in tests.items():
        logger.info(f"Processing test file {test_fname} with {len(entries)} queries")
        results = []
        count = 0
        for e in entries:
            if count == 10:
                break
            if args.limit and count >= args.limit:
                break
            candidate_id = e["candidate"]
            caption = e["caption"] or ""

            # locate candidate image in images_dir
            candidate_path = None
            for ext in ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]:
                p = os.path.join(retriever.images_dir, f"{candidate_id}.{ext}")
                if os.path.exists(p):
                    candidate_path = p
                    break

            if not candidate_path:
                logger.warning(f"Candidate image for {candidate_id} not found in {retriever.images_dir} - skipping.")
                continue

            # Create query embedding (candidate image + caption)
            try:
                query_vec = retriever.encode_candidate_query(candidate_path, caption)
            except Exception as ex:
                logger.warning(f"Failed to encode candidate {candidate_id}: {ex}")
                continue

            # Search
            distances, indices, retrieved_ids = retriever.search(query_vec, k=args.k)
            predicted_id = retrieved_ids[0] if retrieved_ids else None

            # find path for predicted id (search in aligned df)
            predicted_path = None
            if predicted_id is not None:
                # df may contain many entries with same image_id; pick first matching path
                matches = retriever.df[retriever.df["image_id"].astype(str).str.lower() == str(predicted_id).lower()]
                if not matches.empty:
                    predicted_path = matches.iloc[0]["image_path"]
                else:
                    # fallback: try to find file in images dir
                    predicted_path = retriever._find_image(predicted_id)

            results.append({
                "candidate_id": candidate_id,
                "candidate_path": candidate_path,
                "caption": caption,
                "predicted_target_id": str(predicted_id) if predicted_id is not None else None,
                "predicted_target_path": predicted_path
            })

            count += 1

        # Visualize for this test file
        out_base = Path(args.dataset_dir) / f"results_{Path(test_fname).stem}.png"
        retriever.visualize_results(results, out_path=str(out_base))
        logger.info(f"Completed processing {test_fname}")


if __name__ == "__main__":
    main()
