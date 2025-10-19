#!/usr/bin/env python3
"""
StyleFit Streamlit Application
=============================

Advanced Streamlit interface for StyleFit multimodal fashion recommendation system.
Features:
- Interactive text-based queries
- Real-time recommendation generation
- Performance metrics display
- Beautiful UI with custom styling
- Caching for improved performance

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
import streamlit as st
from typing import List, Dict, Optional
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="StyleFit - Fashion Recommendation",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: bold;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .recommendation-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        transition: transform 0.2s ease;
        color: #333333 !important;
    }
    
    .recommendation-card h4 {
        color: #333333 !important;
    }
    
    .recommendation-card p {
        color: #333333 !important;
    }
    
    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .query-input {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        font-size: 1.1rem;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    .loading-spinner {
        text-align: center;
        padding: 2rem;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        border-top: 1px solid #e9ecef;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

class StyleFitApp:
    """Main StyleFit Streamlit application class"""
    
    def __init__(self):
        self.dataset_dir = "data"
        self.images_dir = os.path.join(self.dataset_dir, "images", "pics")
        self.csv_path = os.path.join(self.dataset_dir, "aligned_dataset.csv")
        self.embeds_path = os.path.join(self.dataset_dir, "candidate_embeds.npy")
        self.index_path = os.path.join(self.dataset_dir, "image_index.faiss")
        
        # Initialize session state
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'recommendations' not in st.session_state:
            st.session_state.recommendations = []
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
    
    @st.cache_resource
    def load_model(_self):
        """Load CLIP model with caching"""
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            return model, processor, device
        except Exception as e:
            st.error(f"Failed to load CLIP model: {e}")
            return None, None, None
    
    @st.cache_data
    def load_dataset(_self):
        """Load dataset with caching"""
        try:
            df = pd.read_csv(_self.csv_path)
            
            def parse_captions(x):
                try:
                    if isinstance(x, str):
                        # Handle string that contains Python list representation
                        if x.startswith('[') and x.endswith(']'):
                            # Use eval to parse Python list string (safe in this context)
                            return eval(x)
                        else:
                            return [x]
                    return x if isinstance(x, list) else []
                except:
                    return []

            df['captions'] = df['captions'].apply(parse_captions)
            return df
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            return None
    
    @st.cache_data
    def load_embeddings_and_index(_self):
        """Load embeddings and FAISS index with caching"""
        try:
            if os.path.exists(_self.embeds_path) and os.path.exists(_self.index_path):
                embeddings = np.load(_self.embeds_path)
                index = faiss.read_index(_self.index_path)
                return embeddings, index
            else:
                st.warning("Embeddings or index not found. Please run main.py first.")
                return None, None
        except Exception as e:
            st.error(f"Failed to load embeddings/index: {e}")
            return None, None
    
    def load_image_safe(self, path: str) -> Optional[Image.Image]:
        """Safely load image with error handling"""
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")
            return None
    
    def generate_text_embedding(self, text: str, model, processor, device):
        """Generate text embedding for query"""
        try:
            inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                features = model.get_text_features(**inputs)
                features = F.normalize(features, p=2, dim=1)
            return features.cpu().numpy()
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return None
    
    def search_recommendations(self, query: str, k: int = 5) -> List[Dict]:
        """Search for recommendations using text query"""
        model, processor, device = self.load_model()
        if model is None:
            return []
        
        df = self.load_dataset()
        if df is None:
            return []
        
        embeddings, index = self.load_embeddings_and_index()
        if index is None:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.generate_text_embedding(query, model, processor, device)
            if query_embedding is None:
                return []
            
            # Search
            distances, indices = index.search(query_embedding, k)
            
            recommendations = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(df):
                    row = df.iloc[idx]
                    
                    # Use the image_path from the dataset (now points to target images)
                    img_path = row.get('image_path', '')
                    
                    recommendations.append({
                        'rank': i + 1,
                        'image_id': row['image_id'],  # This is the target image ID
                        'candidate_id': row.get('candidate', ''),  # Keep for reference
                        'captions': row['captions'],
                        'category': row.get('category', 'unknown'),
                        'similarity_score': float(dist),
                        'image_path': img_path
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error searching recommendations: {e}")
            return []
    
    def display_recommendations(self, recommendations: List[Dict]):
        """Display recommendations in a beautiful format"""
        if not recommendations:
            st.warning("No recommendations found.")
            return
        
        st.markdown("### 🎯 Recommendations")
        
        for rec in recommendations:
            with st.container():
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4 style="color: #333333;">Rank #{rec['rank']} | Similarity: {rec['similarity_score']:.4f}</h4>
                    <p style="color: #333333;"><strong>Image ID:</strong> {rec['image_id']}</p>
                    <p style="color: #333333;"><strong>Category:</strong> {rec['category']}</p>
                    <p style="color: #333333;"><strong>Captions:</strong> {', '.join(rec['captions']) if rec['captions'] else 'No captions'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display image if available
                if rec['image_path'] and os.path.exists(rec['image_path']):
                    try:
                        image = self.load_image_safe(rec['image_path'])
                        if image:
                            st.image(image, caption=f"Rank {rec['rank']}: {rec['image_id']}", 
                                   width='stretch')
                    except Exception as e:
                        st.warning(f"Failed to load image: {e}")
                else:
                    st.warning(f"Image not found for {rec['image_id']}")
                
                st.markdown("---")
    
    def display_metrics(self, recommendations: List[Dict]):
        """Display performance metrics"""
        if not recommendations:
            return
        
        st.markdown("### 📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(recommendations)}</div>
                <div class="metric-label">Recommendations</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_score = np.mean([rec['similarity_score'] for rec in recommendations])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_score:.3f}</div>
                <div class="metric-label">Avg Similarity</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            categories = [rec['category'] for rec in recommendations]
            unique_categories = len(set(categories))
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{unique_categories}</div>
                <div class="metric-label">Categories</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            best_score = max([rec['similarity_score'] for rec in recommendations])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{best_score:.3f}</div>
                <div class="metric-label">Best Match</div>
            </div>
            """, unsafe_allow_html=True)
    
    def run(self):
        """Main application runner"""
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>👗 StyleFit</h1>
            <p>Multimodal Fashion Recommendation System</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.markdown("### 🔧 Settings")
            
            # Model status
            model, processor, device = self.load_model()
            if model is not None:
                st.success(f"✅ Model loaded on {device}")
            else:
                st.error("❌ Model failed to load")
                return
            
            # Dataset status
            df = self.load_dataset()
            if df is not None:
                st.success(f"✅ Dataset loaded: {len(df)} items")
            else:
                st.error("❌ Dataset failed to load")
                return
            
            # Embeddings status
            embeddings, index = self.load_embeddings_and_index()
            if index is not None:
                st.success(f"✅ Index loaded: {index.ntotal} vectors")
            else:
                st.error("❌ Embeddings/Index not found")
                return
            
            st.markdown("---")
            
            # Query history
            if st.session_state.query_history:
                st.markdown("### 📝 Recent Queries")
                for i, query in enumerate(st.session_state.query_history[-5:]):
                    if st.button(f"🔍 {query[:30]}...", key=f"history_{i}"):
                        st.session_state.current_query = query
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🎯 Describe Your Ideal Outfit")
            
            # Query input
            query = st.text_input(
                "Enter your style preferences:",
                placeholder="e.g., 'a black elegant dress for evening wear'",
                key="query_input"
            )
            
            # Parameters
            col_k, col_btn = st.columns([1, 2])
            with col_k:
                k = st.slider("Number of recommendations:", 1, 20, 5)
            
            with col_btn:
                search_button = st.button("🔍 Find Recommendations", type="primary")
        
        with col2:
            st.markdown("### 💡 Tips")
            st.markdown("""
            - Be specific about colors, styles, and occasions
            - Use descriptive adjectives (elegant, casual, formal)
            - Mention fabric types (cotton, silk, denim)
            - Include seasonal preferences
            """)
        
        # Search and display results
        if search_button and query:
            with st.spinner("🔍 Searching for recommendations..."):
                start_time = time.time()
                recommendations = self.search_recommendations(query, k)
                search_time = time.time() - start_time
                
                if recommendations:
                    # Add to query history
                    st.session_state.query_history.append(query)
                    st.session_state.recommendations = recommendations
                    
                    # Display success message
                    st.markdown(f"""
                    <div class="success-message">
                        ✅ Found {len(recommendations)} recommendations in {search_time:.2f}s
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display metrics
                    self.display_metrics(recommendations)
                    
                    # Display recommendations
                    self.display_recommendations(recommendations)
                    
                else:
                    st.markdown("""
                    <div class="error-message">
                        ❌ No recommendations found. Try a different query.
                    </div>
                    """, unsafe_allow_html=True)
        
        elif search_button and not query:
            st.warning("⚠️ Please enter a query to search for recommendations.")
        
        # Footer
        st.markdown("""
        <div class="footer">
            <p>StyleFit - Multimodal Fashion Recommendation System</p>
            <p>Powered by CLIP and FAISS</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit app"""
    app = StyleFitApp()
    app.run()

if __name__ == "__main__":
    main()