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
from models.sae_baseline import TiedGaitSAE
from data.scripts.load_data import load_gait_data
from data.scripts.preprocess import preprocess_gait_signals

# For Colab
# from google.colab import drive
# drive.mount('/content/drive')
# DATA_ROOT = '/content/drive/MyDrive/GaitSAE'

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

def load_model(model_path, input_dim, encoding_dim, sparsity_param):
    """Load the trained model"""
    model = TiedGaitSAE(input_dim, encoding_dim, sparsity_param)
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

def analyze_feature_activations(model, data, feature_idx, output_dir):
    """Analyze how a specific feature activates across different gait patterns"""
    with torch.no_grad():
        activations = model.encode(torch.FloatTensor(data))
        activations = activations.cpu().numpy()
    
    plt.figure(figsize=(10, 4))
    plt.hist(activations[:, feature_idx], bins=50)
    plt.title(f'Activation Distribution for Feature {feature_idx}')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    # For Colab
    # plt.savefig(f'/content/drive/MyDrive/GaitSAE/feature_{feature_idx}_activations.png')
    # For local
    plt.savefig(os.path.join(output_dir, f'feature_{feature_idx}_activations.png'))
    plt.close()
    
    return activations[:, feature_idx]

def analyze_feature_metadata_correlations(model, data_dict, processed_data, feature_idx, output_dir):
    """Analyze how a specific feature correlates with metadata attributes"""
    # Get metadata
    metadata = data_dict['GRF_metadata']
    walking_speed = data_dict['GRF_walking_speed']
    
    # Get the first processed data key and its data
    data_key = next(iter(processed_data.keys()))
    processed_data_values = processed_data[data_key]
    
    # Get activations for the feature
    with torch.no_grad():
        activations = model.encode(torch.FloatTensor(processed_data_values))
        activations = activations.cpu().numpy()
    
    # Get the corresponding metadata indices
    # The processed data is from PRO files, so we need to match with the same indices
    data_df = data_dict[data_key]
    subject_ids = data_df['SUBJECT_ID'].values
    session_ids = data_df['SESSION_ID'].values
    trial_ids = data_df['TRIAL_ID'].values
    
    # Create a DataFrame for analysis
    analysis_df = pd.DataFrame({
        'subject_id': subject_ids,
        'session_id': session_ids,
        'trial_id': trial_ids,
        'feature_activation': activations[:, feature_idx]
    })
    
    # Merge with walking speed data
    analysis_df = analysis_df.merge(
        walking_speed[['SUBJECT_ID', 'SESSION_ID', 'TRIAL_ID', 'WALKING_SPEED']],
        left_on=['subject_id', 'session_id', 'trial_id'],
        right_on=['SUBJECT_ID', 'SESSION_ID', 'TRIAL_ID']
    )
    
    # Merge with metadata
    analysis_df = analysis_df.merge(
        metadata[['SUBJECT_ID', 'AGE', 'HEIGHT', 'BODY_WEIGHT', 'SEX']],
        left_on='subject_id',
        right_on='SUBJECT_ID'
    )
    
    # Plot correlations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Walking speed vs activation
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=analysis_df, x='WALKING_SPEED', y='feature_activation')
    plt.title(f'Feature {feature_idx} Activation vs Walking Speed')
    
    # Plot 2: Age vs activation
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=analysis_df, x='AGE', y='feature_activation')
    plt.title(f'Feature {feature_idx} Activation vs Age')
    
    # Plot 3: Height vs activation
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=analysis_df, x='HEIGHT', y='feature_activation')
    plt.title(f'Feature {feature_idx} Activation vs Height')
    
    # Plot 4: Weight vs activation
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=analysis_df, x='BODY_WEIGHT', y='feature_activation')
    plt.title(f'Feature {feature_idx} Activation vs Weight')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'feature_{feature_idx}_metadata_correlations.png'))
    plt.close()
    
    # Calculate correlations
    correlations = analysis_df[['WALKING_SPEED', 'AGE', 'HEIGHT', 'BODY_WEIGHT', 'feature_activation']].corr()['feature_activation'].drop('feature_activation')
    print(f"\nFeature {feature_idx} correlations with metadata:")
    print(correlations)
    
    # Analyze gender differences
    gender_means = analysis_df.groupby('SEX')['feature_activation'].mean()
    print(f"\nFeature {feature_idx} mean activation by gender:")
    print(gender_means)
    
    return correlations, gender_means

def main():
    # Load data
    print("Loading data...")
    data_dict = load_gait_data(use_gaitrec=False)
    processed_data = preprocess_gait_signals(data_dict)
    
    # Get input dimension from processed data
    input_dim = next(iter(processed_data.values())).shape[1]
    encoding_dim = 100  # Should match your training parameters
    sparsity_param = 0.05  # Should match your training parameters
    
    # Load model
    print("Loading model...")
    model_path = os.path.join(MODEL_DIR, 'gait_sae_disent.pth') # change to whatever model you want to analyze
    model_name = os.path.splitext(os.path.basename(model_path))[0]  # Extract model name
    model = load_model(model_path, input_dim, encoding_dim, sparsity_param)
    
    # Update MODEL_DIR to include model name
    output_dir = os.path.join(MODEL_DIR, f'sae_analysis_{model_name}')
    os.makedirs(output_dir, exist_ok=True)

    # Visualize weights
    print("Visualizing weights...")
    visualize_weights(model, output_dir)
    
    # Analyze specific features
    print("Analyzing feature activations...")
    for feature_idx in range(20):  # Analyze first 3 features
        # Basic activation analysis
        activations = analyze_feature_activations(model, next(iter(processed_data.values())), feature_idx, output_dir)
        print(f"Feature {feature_idx} - Mean activation: {np.mean(activations):.4f}, "
              f"Std: {np.std(activations):.4f}, "
              f"Sparsity: {np.mean(activations > 0):.4f}")
        
        # Metadata correlation analysis
        print(f"\nAnalyzing metadata correlations for feature {feature_idx}...")
        correlations, gender_means = analyze_feature_metadata_correlations(
            model, data_dict, processed_data, feature_idx, output_dir
        )

if __name__ == "__main__":
    main()
