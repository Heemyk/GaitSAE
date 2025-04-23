import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pywt

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.sparse_vae import SparseVAE
from data.scripts.load_data import load_gait_data
from data.scripts.preprocess import preprocess_gait_signals

# Constants
DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(DATA_ROOT, 'train', 'model_data')

def load_model(model_path, input_dim, latent_dim, sparsity_param):
    """Load the trained SparseVAE model"""
    model = SparseVAE(input_dim, latent_dim, sparsity_param)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def get_latent_representations(model, data):
    """Get latent representations for all data points using the analyze method"""
    with torch.no_grad():
        x_recon, mu, log_var, z = model.analyze(torch.FloatTensor(data))
        return z.cpu().numpy()

def visualize_latent_space(latent_representations, metadata, walking_speed, output_path):
    """Visualize latent space using t-SNE and color by metadata attributes"""
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_representations)
    
    # Create subplots for different metadata attributes
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot for each metadata attribute
    attributes = ['AGE', 'HEIGHT', 'BODY_WEIGHT', 'WALKING_SPEED']
    for i, attr in enumerate(attributes):
        info = walking_speed if attr == "WALKING_SPEED" else metadata[attr]
        scatter = axes[i].scatter(latent_2d[:, 0], latent_2d[:, 1], 
                                c=info, cmap='viridis')
        axes[i].set_title(f'Latent Space Colored by {attr}')
        axes[i].set_xlabel('t-SNE 1')
        axes[i].set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Load data
    print("Loading data...")
    data_dict = load_gait_data(use_gaitrec=False)
    processed_data = preprocess_gait_signals(data_dict)
    
    # Align metadata and walking speeds
    grf_metadata_df = data_dict['GRF_metadata']
    grf_walking_speed_df = data_dict['GRF_walking_speed']
    aligned_df = pd.merge(grf_walking_speed_df, grf_metadata_df, on='SESSION_ID', how='left')
    aligned_df.drop(columns=['DATASET_ID_y', 'SUBJECT_ID_y'], inplace=True)
    aligned_df.rename(columns={'DATASET_ID_x': 'DATASET_ID', 'SUBJECT_ID_x': 'SUBJECT_ID'}, inplace=True)
    sample_metadata = aligned_df[['SUBJECT_ID', 'AGE', 'SEX', 'HEIGHT', 'BODY_WEIGHT', 'SHOE_SIZE', 'AFFECTED_SIDE', 'TRAIN', 'TRAIN_BALANCED', 'TEST']]
    sample_walking_speeds = aligned_df['WALKING_SPEED']

    # Get input dimension from processed data
    input_dim = next(iter(processed_data.values())).shape[1]
    latent_dim = 100  # Should match your training parameters
    sparsity_param = 0.05  # Should match your training parameters
    
    # Load model
    print("Loading model...")
    model_path = os.path.join(MODEL_DIR, 'gait_sae_vae.pth')
    model_name = os.path.splitext(os.path.basename(model_path))[0]  # Extract model name
    model = load_model(model_path, input_dim, latent_dim, sparsity_param)

    # Update OUTPUT_DIR to include model name
    output_dir = os.path.join(MODEL_DIR, f'latent_space_analysis_{model_name}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get latent representations
    print("Getting latent representations...")
    data_key = next(iter(processed_data.keys()))
    latent_representations = get_latent_representations(model, processed_data[data_key])
    
    # Visualize latent space
    print("Visualizing latent space...")
    visualize_latent_space(latent_representations, sample_metadata, sample_walking_speeds,
                         os.path.join(output_dir, 'latent_space_visualization.png'))

if __name__ == "__main__":
    main()