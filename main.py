#!/usr/bin/env python3
"""
StyleFit Main Training and Evaluation Pipeline
==============================================

This script handles:
1. CLIP model loading and configuration
2. Efficient embedding generation with caching
3. FAISS index construction and optimization
4. Evaluation metrics computation (Precision@k, Recall@k, Hit Rate@k, nDCG@k)
5. Model performance analysis

Author: StyleFit Team
"""

import os
import json
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple, Optional
import argparse
from pathlib import Path
import pickle
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StyleFitModel:
    """Main StyleFit model for multimodal fashion recommendation"""
    
    def __init__(self, dataset_dir: str = "data", model_name: str = "openai/clip-vit-base-patch32"):
        self.dataset_dir = dataset_dir
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Paths
        self.csv_path = os.path.join(dataset_dir, "aligned_dataset.csv")
        self.images_dir = os.path.join(dataset_dir, "images", "pics")
        self.embeds_path = os.path.join(dataset_dir, "candidate_embeds.npy")
        self.index_path = os.path.join(dataset_dir, "image_index.faiss")
        self.metadata_path = os.path.join(dataset_dir, "model_metadata.pkl")
        
        # Initialize model
        self._load_model()
        self._load_dataset()
    
    def _load_model(self):
        """Load CLIP model and processor"""
        logger.info(f"Loading CLIP model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()
            logger.info("CLIP model loaded successfully")
        except Exception as e:
                    logger.error(f"Failed to load CLIP model: {e}")
                    raise
    
    def _load_dataset(self):
        """Load and preprocess dataset"""
        logger.info(f"Loading dataset from {self.csv_path}")
        
        try:
            self.df = pd.read_csv(self.csv_path)
            
            # Parse captions safely
            def parse_captions(x):
                try:
                    if isinstance(x, str):
                        # Handle string that contains Python list representation
                        if x.startswith('[') and x.endswith(']'):
                            return eval(x)
                        else:
                            return [x]
                    return x if isinstance(x, list) else []
                except:
                    return []
            
            self.df['captions'] = self.df['captions'].apply(parse_captions)
            logger.info(f"Loaded {len(self.df)} entries from dataset")
            
        except Exception as e:
                    logger.error(f"Failed to load dataset: {e}")
                    raise
    
    def load_image_safe(self, path: str) -> Optional[Image.Image]:
        """Safely load image with error handling"""
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")
            return None
    
    def generate_embeddings(self, batch_size: int = 32, use_cache: bool = True) -> np.ndarray:
        """Generate multimodal embeddings with caching"""
        
        # Check if cached embeddings exist
        if use_cache and os.path.exists(self.embeds_path):
            logger.info("Loading cached embeddings...")
            try:
                embeddings = np.load(self.embeds_path)
                logger.info(f"Loaded {embeddings.shape[0]} cached embeddings")
                return embeddings
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}")
        
        logger.info("Generating embeddings...")
        all_embeds = []
        valid_indices = []
        
        # Process in batches
        num_batches = (len(self.df) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Generating embeddings"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(self.df))
            batch_df = self.df.iloc[start_idx:end_idx]
            
            batch_images = []
            batch_texts = []
            batch_valid_indices = []
            
            for idx, (_, row) in enumerate(batch_df.iterrows()):
                # Load TARGET image (now using image_id which is the target)
                img_path = row.get('image_path', os.path.join(self.images_dir, f"{row['image_id']}.png"))
                image = self.load_image_safe(img_path)
                
                if image is None:
                    continue
                
                # Prepare text (captions belong to the target image)
                captions = row['captions']
                if not captions:
                    continue
                
                # Use first caption or combine all
                text = captions[0] if len(captions) == 1 else " ".join(captions)
                
                batch_images.append(image)
                batch_texts.append(text)
                batch_valid_indices.append(start_idx + idx)
            
            if not batch_images:
                continue
            
            try:
                # Process batch
                with torch.no_grad():
                    # Process images
                    image_inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                    image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
                    image_features = self.model.get_image_features(**image_inputs)
                    
                    # Process texts
                    text_inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
                    text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                    text_features = self.model.get_text_features(**text_inputs)
                    
                    # Combine features (multimodal fusion)
                    combined_features = (image_features + text_features) / 2
                    
                    # Normalize
                    combined_features = F.normalize(combined_features, p=2, dim=1)
                    
                    all_embeds.append(combined_features.cpu().numpy())
                    valid_indices.extend(batch_valid_indices)
                
            except Exception as e:
                logger.warning(f"Error processing batch {batch_idx}: {e}")
                continue
        
        if not all_embeds:
            raise ValueError("No valid embeddings generated")
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeds)
        
        # Save embeddings
        if use_cache:
            np.save(self.embeds_path, embeddings)
            logger.info(f"Saved embeddings to {self.embeds_path}")
        
        logger.info(f"Generated {embeddings.shape[0]} embeddings")
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray, index_type: str = "cosine") -> faiss.Index:
        """Build FAISS index for efficient similarity search"""
        logger.info(f"Building FAISS index with {index_type} similarity...")
        
        dimension = embeddings.shape[1]
        
        if index_type == "cosine":
            # For cosine similarity, we need to normalize embeddings
            embeddings = embeddings.astype('float32')
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        else:
            # For L2 distance
            index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        index.add(embeddings)
        
        # Save index
        faiss.write_index(index, self.index_path)
        logger.info(f"FAISS index built and saved to {self.index_path}")
        
        return index
    
    def load_faiss_index(self) -> faiss.Index:
        """Load existing FAISS index"""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        
        index = faiss.read_index(self.index_path)
        logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
        return index

    def search_similar(self, query_text: str, k: int = 5, index: Optional[faiss.Index] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar items using text query"""
        if index is None:
            index = self.load_faiss_index()
        
        # Generate query embedding
        with torch.no_grad():
            query_inputs = self.processor(text=query_text, return_tensors="pt", padding=True, truncation=True)
            query_inputs = {k: v.to(self.device) for k, v in query_inputs.items()}
            query_features = self.model.get_text_features(**query_inputs)
            query_features = F.normalize(query_features, p=2, dim=1)
        
        # Search
        distances, indices = index.search(query_features.cpu().numpy(), k)
        return distances, indices

    def evaluate_metrics(self, test_queries: List[str], ground_truth: List[List[int]], k_values: List[int] = [1, 5, 10]) -> Dict:
        """Evaluate model performance with various metrics"""
        logger.info("Evaluating model performance...")
        
        index = self.load_faiss_index()
        results = {}
        
        for k in k_values:
            precision_scores = []
            recall_scores = []
            hit_rates = []
            ndcg_scores = []
            
            for query, gt in zip(test_queries, ground_truth):
                try:
                    distances, indices = self.search_similar(query, k=k, index=index)
                    retrieved = indices[0].tolist()
                    
                    # Calculate metrics
                    precision = len(set(retrieved) & set(gt)) / k
                    recall = len(set(retrieved) & set(gt)) / len(gt) if gt else 0
                    hit_rate = 1 if len(set(retrieved) & set(gt)) > 0 else 0
                    
                    # nDCG calculation
                    ndcg = self._calculate_ndcg(retrieved, gt, k)
                    
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    hit_rates.append(hit_rate)
                    ndcg_scores.append(ndcg)
                
                except Exception as e:
                        logger.warning(f"Error evaluating query: {e}")
                continue
        
            results[f'precision@{k}'] = np.mean(precision_scores)
            results[f'recall@{k}'] = np.mean(recall_scores)
            results[f'hit_rate@{k}'] = np.mean(hit_rates)
            results[f'ndcg@{k}'] = np.mean(ndcg_scores)
        
        return results
    
    def _calculate_ndcg(self, retrieved: List[int], ground_truth: List[int], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        if not ground_truth:
            return 0.0
        
        # DCG calculation
        dcg = 0.0
        for i, item in enumerate(retrieved[:k]):
            if item in ground_truth:
                dcg += 1.0 / np.log2(i + 2)
        
        # IDCG calculation (ideal DCG)
        idcg = 0.0
        for i in range(min(len(ground_truth), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def generate_recommendations(self, query: str, k: int = 5) -> List[Dict]:
        """Generate recommendations for a text query"""
        logger.info(f"Generating recommendations for: {query}")
        
        try:
            distances, indices = self.search_similar(query, k=k)
            
            recommendations = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.df):
                    row = self.df.iloc[idx]
                    recommendations.append({
                        'rank': i + 1,
                        'image_id': row['image_id'],  # This is now the target image ID
                        'candidate_id': row.get('candidate', ''),  # Keep for reference
                        'captions': row['captions'],
                        'category': row.get('category', 'unknown'),
                        'similarity_score': float(dist),
                        'image_path': row.get('image_path', '')
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def save_metadata(self, results: Dict):
        """Save model metadata and results"""
        metadata = {
            'model_name': self.model_name,
            'dataset_size': len(self.df),
            'embedding_dim': self.model.config.projection_dim,
            'device': str(self.device),
            'timestamp': time.time(),
            'results': results
        }
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Metadata saved to {self.metadata_path}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='StyleFit Main Pipeline')
    parser.add_argument('--dataset_dir', default='data', help='Dataset directory path')
    parser.add_argument('--model_name', default='openai/clip-vit-base-patch32', help='CLIP model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--k_values', nargs='+', type=int, default=[1, 5, 10], help='K values for evaluation')
    parser.add_argument('--no_cache', action='store_true', help='Disable embedding caching')
    
    args = parser.parse_args()
    
    try:
        # Initialize model
        model = StyleFitModel(args.dataset_dir, args.model_name)
        
        # Generate embeddings
        embeddings = model.generate_embeddings(batch_size=args.batch_size, use_cache=not args.no_cache)

# Build FAISS index
        index = model.build_faiss_index(embeddings)
        
        # Example evaluation (you can add your own test queries)
        test_queries = [
            "a black dress",
            "casual shirt",
            "formal outfit",
            "summer clothing",
            "elegant evening wear"
        ]
        
        # For demonstration, create dummy ground truth
        ground_truth = [[i] for i in range(len(test_queries))]
        
        # Evaluate metrics
        results = model.evaluate_metrics(test_queries, ground_truth, args.k_values)
        
        # Print results
        print("\nEvaluation Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save metadata
        model.save_metadata(results)
        
        # Example recommendation
        print("\nExample Recommendation:")
        recommendations = model.generate_recommendations("a stylish black dress", k=3)
        for rec in recommendations:
            print(f"  Rank {rec['rank']}: {rec['candidate_id']} (Score: {rec['similarity_score']:.4f})")
        
        print("\nStyleFit model training and evaluation completed!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()