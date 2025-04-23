import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os

# For Colab
# from google.colab import drive
# drive.mount('/content/drive')
# sys.path.append('/content/drive/MyDrive/GaitSAE')  # Add your code to Python path

# For local
from data.scripts.load_data import load_gait_data
from data.scripts.preprocess import preprocess_gait_signals
from models.sae_baseline import TiedGaitSAE
from models.sparse_vae import SparseVAE

# For Colab
# DATA_ROOT = '/content/drive/MyDrive/GaitSAE'
# For local
DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_dataloader(processed_data, batch_size=32):
    """
    Create a DataLoader from the processed data
    """
    # Combine all processed signals into a single dataset
    all_signals = []
    for key, data in processed_data.items():
        all_signals.append(data)
    
    # Stack all signals
    X = np.concatenate(all_signals, axis=0)
    
    # Convert to torch tensors
    X = torch.FloatTensor(X)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, X.shape[1]  # Return input dimension

def train_sae(model, data_loader, optimizer, epochs=100, device='cuda'):
    """
    Train the sparse autoencoder
    """
    model = model.to(device)
    model.train()
    
    # Store losses for plotting
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            # Move batch to device
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            # For SparseVAE, unpack all four outputs
            if isinstance(model, SparseVAE):
                recon, mu, log_var, z = model(x)
            else:
                recon, activations = model(x)  # For normal SAE

            # Calculate loss
            if isinstance(model, SparseVAE):
                loss = model.loss_function(x, recon, mu, log_var, z)
            else:
                loss = model.loss_function(x, recon, activations)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average loss for this epoch
        avg_loss = total_loss/len(data_loader)
        losses.append(avg_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
    
    return losses

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check for available device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data_dict = load_gait_data(use_gaitrec=False)
    
    # Preprocess data
    print("Preprocessing data...")
    processed_data = preprocess_gait_signals(data_dict)
    
    # Create dataloader
    print("Creating dataloader...")
    dataloader, input_dim = create_dataloader(processed_data, batch_size=32)
    
    # Initialize model
    print("Initializing model...")
    encoding_dim = 100  # Number of features to learn
    sparsity_param = 0.05  # Target sparsity
    # model = TiedGaitSAE(input_dim, encoding_dim, sparsity_param)
    model = SparseVAE(input_dim, encoding_dim, sparsity_param)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print("Training model...")
    losses = train_sae(model, dataloader, optimizer, epochs=100, device=device)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # For Colab
    # plt.savefig('/content/drive/MyDrive/GaitSAE/training_loss.png')
    # For local
    plt.savefig(os.path.join(DATA_ROOT, 'training_loss.png'))
    plt.close()
    
    # Save model
    # For Colab
    # torch.save(model.state_dict(), '/content/drive/MyDrive/GaitSAE/gait_sae.pth')
    # For local
    torch.save(model.state_dict(), os.path.join(DATA_ROOT, 'gait_sae.pth'))
    print("Model saved to gait_sae.pth")

if __name__ == "__main__":
    main()
