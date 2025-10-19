# #!/usr/bin/env python3
# """
# StyleFit Data Preprocessing Pipeline
# ====================================

# This script processes the fashion dataset by:
# 1. Loading caption data from JSON files
# 2. Aligning captions with available images
# 3. Creating a clean, aligned dataset for training
# 4. Generating dataset statistics and validation

# Author: StyleFit Team
# """

# import os
# import json
# import pandas as pd
# from pathlib import Path
# import logging
# from PIL import Image
# import matplotlib.pyplot as plt
# import random
# from typing import List, Dict, Tuple, Optional
# import argparse

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class StyleFitPreprocessor:
#     """Handles preprocessing of StyleFit fashion dataset"""
    
#     def __init__(self, dataset_dir: str = "data"):
#         self.dataset_dir = dataset_dir
#         self.captions_dir = os.path.join(dataset_dir, "captions")
#         self.images_dir = os.path.join(dataset_dir, "images", "pics")
#         self.output_csv = os.path.join(dataset_dir, "aligned_dataset.csv")
        
#         # Validate directories
#         self._validate_directories()
    
#     def _validate_directories(self):
#         """Validate that required directories exist"""
#         if not os.path.exists(self.captions_dir):
#             raise FileNotFoundError(f"Captions directory not found: {self.captions_dir}")
#         if not os.path.exists(self.images_dir):
#             raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
#         logger.info(f"Dataset directory: {self.dataset_dir}")
#         logger.info(f"Captions directory: {self.captions_dir}")
#         logger.info(f"Images directory: {self.images_dir}")
    
#     def find_image_path(self, candidate: str, images_dir: str) -> Optional[str]:
#         """Find image path for a candidate ID, checking multiple extensions"""
#         for ext in ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']:
#             path = os.path.join(images_dir, f"{candidate}.{ext}")
#             if os.path.exists(path):
#                 return path
#         return None
    
#     def load_captions(self, directory: str) -> pd.DataFrame:
#         """Load caption data from all JSON files - using TARGET images with their captions"""
#         data = []
#         json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        
#         logger.info(f"Found {len(json_files)} JSON files")
        
#         for file in json_files:
#             file_path = os.path.join(directory, file)
#             logger.info(f"Processing {file}")
            
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     json_data = json.load(f)
                
#                 for entry in json_data:
#                     # Handle different caption formats
#                     captions = entry.get("captions", [])
#                     if isinstance(captions, str):
#                         captions = [captions]
                    
#                     # Use TARGET as the main ID and its captions
#                     target_id = entry.get("target", "")
#                     if target_id and captions:  # Only include entries with both target and captions
#                         data.append({
#                             "file": file,
#                             "image_id": target_id,  # Use target as the main image ID
#                             "candidate": entry.get("candidate", ""),  # Keep for reference
#                             "captions": captions,
#                             "category": self._extract_category(file)
#                         })
                    
#             except Exception as e:
#                 logger.error(f"Error processing {file}: {e}")
#                 continue
        
#         df = pd.DataFrame(data)
#         logger.info(f"Loaded {len(df)} entries from {len(json_files)} JSON files")
#         return df
    
#     def _extract_category(self, filename: str) -> str:
#         """Extract category from filename"""
#         if 'dress' in filename.lower():
#             return 'dress'
#         elif 'shirt' in filename.lower():
#             return 'shirt'
#         elif 'toptee' in filename.lower():
#             return 'toptee'
#         else:
#             return 'unknown'
    
#     def align_data(self, df: pd.DataFrame, images_dir: str) -> Tuple[pd.DataFrame, set]:
#         """Align caption data with available TARGET images"""
#         # Get available image files
#         image_files = [f for f in os.listdir(images_dir) 
#                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#         image_ids = set(Path(f).stem.lower() for f in image_files)
        
#         logger.info(f"Found {len(image_files)} image files")
        
#         # Get target IDs from dataframe (these are the image IDs we want to use)
#         candidate_ids = set(df["candidate"].str.lower().dropna())
#         logger.info(f"Found {len(candidate_ids)} target IDs in captions")
        
#         # Find common IDs between targets and available images
#         common_ids = candidate_ids.intersection(image_ids)
#         logger.info(f"Found {len(common_ids)} common IDs between targets and images")
        
#         # Filter dataframe to only include aligned data
#         df_aligned = df[df["image_id"].str.lower().isin(common_ids)].copy()
        
#         # Add image path column using the target image ID
#         df_aligned['image_path'] = df_aligned['image_id'].apply(
#             lambda x: self.find_image_path(x, images_dir)
#         )
        
#         # Remove rows where image path is None
#         initial_count = len(df_aligned)
#         df_aligned = df_aligned.dropna(subset=['image_path'])
#         final_count = len(df_aligned)
        
#         logger.info(f"Removed {initial_count - final_count} rows with missing images")
        
#         return df_aligned, common_ids
    
#     def validate_images(self, df: pd.DataFrame, sample_size: int = 100) -> None:
#         """Validate that images can be loaded properly"""
#         logger.info(f"Validating {sample_size} random images...")
        
#         sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
#         valid_count = 0
        
#         for idx, row in sample_df.iterrows():
#             try:
#                 img_path = row['image_path']
#                 if img_path and os.path.exists(img_path):
#                     image = Image.open(img_path).convert("RGB")
#                     valid_count += 1
#             except Exception as e:
#                 logger.warning(f"Invalid image {row['candidate']}: {e}")
        
#         logger.info(f"Validated {valid_count}/{len(sample_df)} images successfully")
    
#     def generate_statistics(self, df: pd.DataFrame) -> Dict:
#         """Generate dataset statistics"""
#         stats = {
#             'total_entries': len(df),
#             'unique_images': df['image_id'].nunique(),
#             'unique_candidates': df['candidate'].nunique(),
#             'categories': df['category'].value_counts().to_dict(),
#             'avg_captions_per_entry': df['captions'].apply(len).mean(),
#             'files_processed': df['file'].nunique()
#         }
        
#         # Caption length statistics
#         caption_lengths = df['captions'].apply(lambda x: sum(len(c) for c in x) if x else 0)
#         stats['avg_caption_length'] = caption_lengths.mean()
#         stats['max_caption_length'] = caption_lengths.max()
#         stats['min_caption_length'] = caption_lengths.min()
        
#         return stats
    
#     def visualize_samples(self, df: pd.DataFrame, num_samples: int = 9, save_path: Optional[str] = None):
#         """Visualize sample images with captions"""
#         logger.info(f"Visualizing {num_samples} sample images...")
        
#         sample_df = df.sample(n=min(num_samples, len(df)), random_state=42)
        
#         # Create subplot grid
#         cols = 3
#         rows = (num_samples + cols - 1) // cols
        
#         fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
#         if rows == 1:
#             axes = axes.reshape(1, -1)
        
#         for idx, (_, row) in enumerate(sample_df.iterrows()):
#             row_idx = idx // cols
#             col_idx = idx % cols
            
#             try:
#                 img_path = row['image_path']
#                 if img_path and os.path.exists(img_path):
#                     image = Image.open(img_path).convert("RGB")
#                     axes[row_idx, col_idx].imshow(image)
#                     axes[row_idx, col_idx].axis('off')
                    
#                     # Display caption
#                     caption = row['captions'][0] if row['captions'] else "No caption"
#                     axes[row_idx, col_idx].set_title(f"{row['candidate']}\n{caption[:50]}...", 
#                                                     fontsize=8, pad=5)
#                 else:
#                     axes[row_idx, col_idx].text(0.5, 0.5, f"Image not found\n{row['candidate']}", 
#                                               ha='center', va='center', transform=axes[row_idx, col_idx].transAxes)
#                     axes[row_idx, col_idx].axis('off')
                    
#             except Exception as e:
#                 axes[row_idx, col_idx].text(0.5, 0.5, f"Error loading\n{row['candidate']}\n{str(e)[:20]}", 
#                                           ha='center', va='center', transform=axes[row_idx, col_idx].transAxes)
#                 axes[row_idx, col_idx].axis('off')
        
#         # Hide empty subplots
#         for idx in range(num_samples, rows * cols):
#             row_idx = idx // cols
#             col_idx = idx % cols
#             axes[row_idx, col_idx].axis('off')
        
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, dpi=150, bbox_inches='tight')
#             logger.info(f"Sample visualization saved to {save_path}")
        
#         plt.show()
    
#     def process(self, visualize: bool = True, validate_images: bool = True) -> pd.DataFrame:
#         """Main processing pipeline"""
#         logger.info("Starting StyleFit preprocessing pipeline...")
        
#         # Load captions
#         df = self.load_captions(self.captions_dir)
        
#         # Align with images
#         df_aligned, common_ids = self.align_data(df, self.images_dir)
        
#         # Validate images if requested
#         if validate_images:
#             self.validate_images(df_aligned)
        
#         # Generate statistics
#         stats = self.generate_statistics(df_aligned)
#         logger.info("Dataset Statistics:")
#         for key, value in stats.items():
#             logger.info(f"  {key}: {value}")
        
#         # Save aligned dataset
#         df_aligned.to_csv(self.output_csv, index=False)
#         logger.info(f"Saved aligned dataset to {self.output_csv}")
        
#         # Visualize samples if requested
#         if visualize:
#             self.visualize_samples(df_aligned, num_samples=9)
        
#         logger.info("Preprocessing completed successfully!")
#         return df_aligned

# def main():
#     """Main function with command line interface"""
#     parser = argparse.ArgumentParser(description='StyleFit Data Preprocessing')
#     parser.add_argument('--dataset_dir', default='data', help='Dataset directory path')
#     parser.add_argument('--no_visualize', action='store_true', help='Skip visualization')
#     parser.add_argument('--no_validate', action='store_true', help='Skip image validation')
    
#     args = parser.parse_args()
    
#     try:
#         preprocessor = StyleFitPreprocessor(args.dataset_dir)
#         df_aligned = preprocessor.process(
#             visualize=not args.no_visualize,
#             validate_images=not args.no_validate
#         )
        
#         print(f"\nPreprocessing completed successfully!")
#         print(f"Dataset contains {len(df_aligned)} aligned entries")
#         print(f"Saved to: {preprocessor.output_csv}")
        
#     except Exception as e:
#         logger.error(f"Preprocessing failed: {e}")
#         raise

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
StyleFit Data Preprocessing Pipeline (Simplified + Candidate Display)
=====================================================================

This script processes the StyleFit dataset by:
1. Loading caption data from JSON files
2. Aligning captions with available candidate images
3. Creating a clean, aligned dataset for training
4. Displaying candidate IDs with each sample
"""

import os
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class StyleFitPreprocessor:
    """Simplified preprocessor for StyleFit dataset."""

    def __init__(self, dataset_dir: str = "data"):
        self.dataset_dir = dataset_dir
        self.captions_dir = os.path.join(dataset_dir, "captions")
        self.images_dir = os.path.join(dataset_dir, "images", "pics")
        self.output_csv = os.path.join(dataset_dir, "aligned_dataset.csv")

        if not os.path.exists(self.captions_dir):
            raise FileNotFoundError(f"Missing captions folder: {self.captions_dir}")
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Missing images folder: {self.images_dir}")

    def _find_image(self, image_id: str) -> str | None:
        """Find image path by trying common extensions."""
        for ext in ["png", "jpg", "jpeg"]:
            path = os.path.join(self.images_dir, f"{image_id}.{ext}")
            if os.path.exists(path):
                return path
        return None

    def _extract_category(self, filename: str) -> str:
        """Extract category based on filename keywords."""
        name = filename.lower()
        if "dress" in name:
            return "dress"
        if "shirt" in name:
            return "shirt"
        if "toptee" in name:
            return "toptee"
        return "unknown"

    def load_captions(self) -> pd.DataFrame:
        """Load captions from JSON files (using candidate as main ID)."""
        records = []
        json_files = [f for f in os.listdir(self.captions_dir)if f.endswith("train.json") or f.endswith("val.json")]

        logger.info(f"Found {len(json_files)} caption files")

        for file in json_files:
            file_path = os.path.join(self.captions_dir, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for entry in data:
                    candidate = entry.get("candidate", "")
                    captions = entry.get("captions", [])
                    target = entry.get("target", "")
                    if isinstance(captions, str):
                        captions = [captions]

                    if candidate and captions:
                        records.append({
                            "candidate": candidate,
                            "captions": captions,
                            "file": file,
                            "target": target,
                            "category": self._extract_category(file)

                        })
            except Exception as e:
                logger.warning(f"Error in {file}: {e}")

        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} caption entries")
        return df

    def align_with_images(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align captions with available candidate images."""
        image_files = [f for f in os.listdir(self.images_dir) if f.lower().endswith(("png", "jpg", "jpeg"))]
        available_ids = {Path(f).stem.lower() for f in image_files}

        df = df[df["candidate"].str.lower().isin(available_ids)].copy()
        df["image_path"] = df["candidate"].apply(self._find_image)
        df = df.dropna(subset=["image_path"])

        logger.info(f"Aligned {len(df)} caption entries with available images")
        return df

    def generate_statistics(self, df: pd.DataFrame) -> None:
        """Print basic dataset statistics."""
        stats = {
            "Total entries": len(df),
            "Unique candidates": df["candidate"].nunique(),
            "Categories": df["category"].value_counts().to_dict(),
            "Avg captions per item": round(df["captions"].apply(len).mean(), 2)
        }
        for k, v in stats.items():
            logger.info(f"{k}: {v}")

    def visualize_samples(self, df: pd.DataFrame, n: int = 9):
        """Visualize random samples with candidate IDs."""
        sample_df = df.sample(n=min(n, len(df)), random_state=42)
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        axes = axes.flatten()

        for ax, (_, row) in zip(axes, sample_df.iterrows()):
            img = Image.open(row["image_path"]).convert("RGB")
            ax.imshow(img)
            ax.axis("off")

            caption = row["captions"][0] if row["captions"] else "No caption"
            candidate_id = row["candidate"]

            ax.set_title(f"{candidate_id}\n{caption[:60]}...", fontsize=8, pad=5)

        for ax in axes[len(sample_df):]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def process(self, visualize: bool = True) -> pd.DataFrame:
        """Run the full preprocessing pipeline."""
        df = self.load_captions()
        df = self.align_with_images(df)
        self.generate_statistics(df)

        df.to_csv(self.output_csv, index=False)
        logger.info(f"Saved aligned dataset to {self.output_csv}")

        if visualize:
            self.visualize_samples(df)
        return df

def main():
    parser = argparse.ArgumentParser(description="StyleFit Preprocessing")
    parser.add_argument("--dataset_dir", default="data", help="Path to dataset")
    parser.add_argument("--no_visualize", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    pre = StyleFitPreprocessor(args.dataset_dir)
    df = pre.process(visualize=not args.no_visualize)
    print(f"\n✅ Preprocessing complete! {len(df)} entries saved to {pre.output_csv}")

if __name__ == "__main__":
    main()
