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

from models.sae_baseline import TiedGaitSAE
from data.scripts.load_data import load_gait_data
from data.scripts.preprocess import preprocess_gait_signals

# Constants
DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(DATA_ROOT, 'train', 'model_data')
# OUTPUT_DIR = os.path.join(MODEL_DIR, 'latent_space_analysis')
# os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model(model_path, input_dim, encoding_dim, sparsity_param):
    """Load the trained model"""
    model = TiedGaitSAE(input_dim, encoding_dim, sparsity_param)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def get_latent_representations(model, data):
    """Get latent representations for all data points"""
    with torch.no_grad():
        _, activations = model(torch.FloatTensor(data))
        return activations.cpu().numpy()

def visualize_latent_space(latent_representations, metadata, walking_speed, output_path):
    """Visualize latent space using t-SNE and color by metadata attributes"""
    # Reduce dimensionality using t-SNE
    print("Number of latent vectors:", len(latent_representations))

    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_representations)
    
    # Create subplots for different metadata attributes
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot for each metadata attribute
    attributes = ['AGE', 'HEIGHT', 'BODY_WEIGHT', 'WALKING_SPEED']
    for i, attr in enumerate(attributes):
        info = walking_speed if attr=="WALKING_SPEED" else metadata[attr]
        print(info)
        scatter = axes[i].scatter(latent_2d[:, 0], latent_2d[:, 1], 
                                c=info, cmap='viridis')
        axes[i].set_title(f'Latent Space Colored by {attr}')
        axes[i].set_xlabel('t-SNE 1')
        axes[i].set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_feature_importance(latent_representations, output_dir):
    """Analyze the importance of each feature in the latent space"""
    # Calculate variance explained by each feature
    feature_variances = np.var(latent_representations, axis=0)
    feature_importance = feature_variances / np.sum(feature_variances)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title('Feature Importance in Latent Space')
    plt.xlabel('Feature Index')
    plt.ylabel('Variance Explained')
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()
    
    return feature_importance

def reconstruct_from_feature(model, feature_idx, input_dim):
    """Reconstruct signals using a single feature"""
    # Create a one-hot vector for the feature
    feature_vector = torch.zeros(1, model.encoder_weight.shape[0])
    feature_vector[0, feature_idx] = 1
    
    # Decode the feature
    with torch.no_grad():
        reconstructed = model.decode(feature_vector)
    
    # Inverse wavelet transform
    coeffs = pywt.wavedec(reconstructed.numpy().flatten(), 'db5', level=4)
    reconstructed_signal = pywt.waverec(coeffs, 'db5')
    
    return reconstructed_signal

def visualize_feature_reconstructions(model, input_dim, output_dir, num_features=5):
    """Visualize reconstructions from individual features"""
    plt.figure(figsize=(15, 3*num_features))
    
    for i in range(num_features):
        reconstructed = reconstruct_from_feature(model, i, input_dim)
        
        plt.subplot(num_features, 1, i+1)
        plt.plot(reconstructed)
        plt.title(f'Reconstruction from Feature {i}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_reconstructions.png'))
    plt.close()

def cluster_latent_space(latent_representations, output_dir, n_clusters=5):
    """Cluster the latent space and analyze cluster characteristics"""
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(latent_representations)
    
    # Visualize clusters in 2D using PCA
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_representations)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                         c=cluster_labels, cmap='tab10')
    plt.title('Latent Space Clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(scatter, label='Cluster')
    plt.savefig(os.path.join(output_dir, 'latent_space_clusters.png'))
    plt.close()
    
    return cluster_labels

def main():
    # Load data
    print("Loading data...")
    data_dict = load_gait_data(use_gaitrec=False)
    print(data_dict.keys())

    print(f"Length of metadata_df: {len(data_dict['GRF_metadata'])}")
    print(f"Length of walking_speeds: {len(data_dict['GRF_walking_speed'])}")
    print(f"Length of GRF_F_V_PRO_left: {len(data_dict['GRF_F_V_PRO_left'])}")

    

    processed_data = preprocess_gait_signals(data_dict)
    # processed_data, sample_metadata, sample_walking_speeds = preprocess_gait_signals(data_dict, return_metadata=True)

    # Then align separately if needed:
    grf_metadata_df = data_dict['GRF_metadata']
    grf_walking_speed_df = data_dict['GRF_walking_speed']

    # Do alignment here manually depending on mapping strategy
    aligned_df = pd.merge(grf_walking_speed_df, grf_metadata_df, on='SESSION_ID', how='left')
    # Drop duplicated columns after merge
    aligned_df.drop(columns=['DATASET_ID_y', 'SUBJECT_ID_y'], inplace=True)
    aligned_df.rename(columns={'DATASET_ID_x': 'DATASET_ID', 'SUBJECT_ID_x': 'SUBJECT_ID'}, inplace=True)

    # Check the result to see if the data is correctly aligned
    print(aligned_df.columns)
    print(aligned_df.head())

    # Extract aligned metadata and walking speeds
    sample_metadata = aligned_df[['SUBJECT_ID', 'AGE', 'SEX', 'HEIGHT', 'BODY_WEIGHT', 'SHOE_SIZE', 'AFFECTED_SIDE', 'TRAIN', 'TRAIN_BALANCED', 'TEST']]
    sample_walking_speeds = aligned_df['WALKING_SPEED']


    # Get input dimension from processed data
    input_dim = next(iter(processed_data.values())).shape[1]
    encoding_dim = 100  # Should match your training parameters
    sparsity_param = 0.05  # Should match your training parameters
    
    # Load model
    print("Loading model...")
    model_path = os.path.join(MODEL_DIR, 'gait_sae_disent.pth')
    model_name = os.path.splitext(os.path.basename(model_path))[0]  # Extract model name
    model = load_model(model_path, input_dim, encoding_dim, sparsity_param)

    # Update OUTPUT_DIR to include model name
    output_dir = os.path.join(MODEL_DIR, f'latent_space_analysis_{model_name}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get latent representations
    print("Getting latent representations...")
    data_key = next(iter(processed_data.keys()))
    print(f"Keys in processed_data: {list(processed_data.keys())}")
    print(f"Shape of data under '{data_key}': {processed_data[data_key].shape}")

    
    latent_representations = get_latent_representations(model, processed_data[data_key])
    print(f"Shape of latent representations: {latent_representations.shape}")

    # Compute one latent vector per subject by averaging over their strides
    # latent_representations = []
    # for key in processed_data:
    #     subject_latent = get_latent_representations(model, processed_data[key])
    #     latent_representations.append(subject_latent.mean(axis=0))  # average over strides
    # latent_representations = np.vstack(latent_representations)

    
    # Visualize latent space
    print("Visualizing latent space...")
    # metadata = data_dict['GRF_metadata']
    # walking_speed = data_dict['GRF_walking_speed']
    visualize_latent_space(latent_representations, sample_metadata, sample_walking_speeds,
                         os.path.join(output_dir, 'latent_space_visualization.png'))
    
    # Analyze feature importance
    print("Analyzing feature importance...")
    feature_importance = analyze_feature_importance(latent_representations, output_dir)
    
    # Visualize feature reconstructions
    print("Visualizing feature reconstructions...")
    visualize_feature_reconstructions(model, input_dim, output_dir)
    
    # Cluster latent space
    print("Clustering latent space...")
    cluster_labels = cluster_latent_space(latent_representations, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 