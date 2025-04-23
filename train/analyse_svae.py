import os
import sys
import pandas as pd
import seaborn as sns

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pywt
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.sparse_vae import SparseVAE
from data.scripts.load_data import load_gait_data
from data.scripts.preprocess import preprocess_gait_signals

# For local
DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(DATA_ROOT, 'train', 'model_data')

def analyze_features(model, data):
    """
    Analyze learned features by reconstructing them in the time domain
    """
    # Get encoder weights
    features = model.encoder_weight.detach().cpu().numpy()
    
    # Inverse wavelet transform for visualization
    reconstructed_features = []
    for feature in features:
        # Reshape feature to match wavelet coefficients structure
        coeffs = pywt.wavedec(feature, 'db5', level=4)
        # Reconstruct time domain signal
        reconstructed = pywt.waverec(coeffs, 'db5')
        reconstructed_features.append(reconstructed)
    
    return reconstructed_features

def load_model(model_path, input_dim, latent_dim, sparsity_param):
    """Load the trained SparseVAE model"""
    model = SparseVAE(input_dim, latent_dim, sparsity_param)
    # Load the model with map_location to CPU
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def visualize_weights(model, output_dir, feature_idx=None):
    """Visualize the encoder weights for specific features"""
    weights = model.encoder_weight.detach().cpu().numpy()

    if feature_idx is not None:
        weights = weights[feature_idx]
        plt.figure(figsize=(10, 4))
        plt.plot(weights)
        plt.title(f'Encoder Weights for Feature {feature_idx}')
    else:
        plt.figure(figsize=(15, 10))
        plt.imshow(weights, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title('Encoder Weights Matrix')
    plt.xlabel('Input Dimension')
    plt.ylabel('Feature Dimension')
    plt.tight_layout()

    # Save the figure to the specified output directory
    plt.savefig(os.path.join(output_dir, 'encoder_weights.png'))
    plt.close()

def main():
    # Load data
    print("Loading data...")
    data_dict = load_gait_data(use_gaitrec=False)
    processed_data = preprocess_gait_signals(data_dict)
    
    # Get input dimension from processed data
    input_dim = next(iter(processed_data.values())).shape[1]
    latent_dim = 100  # Should match your training parameters
    sparsity_param = 0.05  # Should match your training parameters
    
    # Load model
    print("Loading model...")
    model_path = os.path.join(MODEL_DIR, 'gait_sae_vae.pth') # change to whatever model you want to analyze
    model_name = os.path.splitext(os.path.basename(model_path))[0]  # Extract model name
    model = load_model(model_path, input_dim, latent_dim, sparsity_param)
    
    # Update MODEL_DIR to include model name
    output_dir = os.path.join(MODEL_DIR, f'svae_analysis_{model_name}')
    os.makedirs(output_dir, exist_ok=True)

    # Visualize weights
    print("Visualizing weights...")
    visualize_weights(model, output_dir)

if __name__ == "__main__":
    main()