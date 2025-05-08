import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# Constants
DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the project root to the Python path
sys.path.append(DATA_ROOT)
MODEL_DIR = os.path.join(DATA_ROOT, 'train', 'model_data')
GRF_OUTPUT_DIR = os.path.join(MODEL_DIR, 'grfs')
os.makedirs(GRF_OUTPUT_DIR, exist_ok=True)

from models.sparse_vae import SparseVAE
from data.scripts.load_data import load_gait_data
from data.scripts.preprocess import preprocess_gait_signals

def load_model(model_path, input_dim, latent_dim, sparsity_param):
    """Load the trained SparseVAE model"""
    model = SparseVAE(input_dim, latent_dim, sparsity_param)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def convert_to_sparse_basis(model, data):
    """Convert all data points to the sparse basis"""
    with torch.no_grad():
        _, _, _, z = model.analyze(torch.FloatTensor(data))
        return z.cpu().numpy()

def plot_top_k_grfs(model, data, sparse_basis, feature_idx, k, output_dir):
    """Plot the top k GRFs for a specific feature, including the reconstructed feature"""
    # Get the top k samples that fire for the feature
    top_k_indices = np.argsort(-sparse_basis[:, feature_idx])[:k]

    # Create a subfolder for the feature
    feature_dir = os.path.join(output_dir, f'feature_{feature_idx}')
    os.makedirs(feature_dir, exist_ok=True)

    for i, idx in enumerate(top_k_indices):
        # Get the original or reconstructed GRF
        original_grf = data[idx]
        reconstructed_grf = model.decode(torch.FloatTensor(sparse_basis[idx:idx+1])).detach().numpy().flatten()

        # Reconstruct the feature itself using the variational interpretation
        feature_vector = torch.zeros(1, sparse_basis.shape[1], dtype=torch.float32)
        feature_vector[0, feature_idx] = float(sparse_basis[idx, feature_idx])
        reconstructed_feature = model.decode(feature_vector).detach().numpy().flatten()

        # For SVAE, sample from the latent space using the mean and variance
        mu = feature_vector
        log_var = torch.zeros_like(mu)  # Assuming log_var is zero for simplicity
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sampled_feature = mu + eps * std
        reconstructed_sampled_feature = model.decode(sampled_feature).detach().numpy().flatten()

        # Plot the GRF
        plt.figure(figsize=(10, 4))
        plt.plot(original_grf, label='Original GRF')
        plt.plot(reconstructed_grf, label='Reconstructed GRF', linestyle='--')
        plt.plot(reconstructed_feature, label='Reconstructed Feature (Mean)', linestyle=':')
        plt.plot(reconstructed_sampled_feature, label='Reconstructed Feature (Sampled)', linestyle='-.')
        plt.title(f'Feature {feature_idx} - Top {i+1} Sample')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(feature_dir, f'sample_{i+1}.png'))
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
    model_path = os.path.join(MODEL_DIR, 'gait_sae_vae.pth')
    model = load_model(model_path, input_dim, latent_dim, sparsity_param)

    # Convert data to sparse basis
    print("Converting data to sparse basis...")
    data_key = next(iter(processed_data.keys()))
    data = processed_data[data_key]
    sparse_basis = convert_to_sparse_basis(model, data)

    # Plot top k GRFs for each feature
    k = 5  # Number of top samples to plot per feature
    print("Plotting top k GRFs for each feature...")
    for feature_idx in range(latent_dim):
        plot_top_k_grfs(model, data, sparse_basis, feature_idx, k, GRF_OUTPUT_DIR)

    print(f"Plots saved to {GRF_OUTPUT_DIR}")

if __name__ == "__main__":
    main()