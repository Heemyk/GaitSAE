import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TiedGaitSAE(nn.Module):
    def __init__(self, input_dim, encoding_dim, sparsity_param=0.05):
        super().__init__()
        
        # Initialize encoder weight
        self.encoder_weight = nn.Parameter(torch.randn(encoding_dim, input_dim) / np.sqrt(input_dim))
        self.encoder_bias = nn.Parameter(torch.zeros(encoding_dim))
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        
        self.sparsity_param = sparsity_param
        
    def encode(self, x):
        # Encoder with tied weights
        return F.relu(F.linear(x, self.encoder_weight, self.encoder_bias))
        
    def decode(self, h):
        # Decoder with tied weights (transpose of encoder weights)
        return F.linear(h, self.encoder_weight.t(), self.decoder_bias)
    
    def forward(self, x):
        h = self.encode(x)
        return self.decode(h), h

    def disentanglement_loss(self, latent_representations):
        cov_matrix = torch.cov(latent_representations.T)
        off_diagonal = cov_matrix - torch.diag(torch.diag(cov_matrix))
        return torch.sum(off_diagonal ** 2)

    def loss_function(self, x, x_recon, activations):
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)
        
        # Sparsity loss (KL divergence)
        rho_hat = torch.mean(activations, dim=0)
        sparsity_loss = torch.sum(
            self.sparsity_param * torch.log(self.sparsity_param / rho_hat) +
            (1 - self.sparsity_param) * torch.log((1 - self.sparsity_param) / (1 - rho_hat))
        )
        
        # L1 regularization for interpretability
        l1_loss = torch.mean(torch.abs(activations))

        # Optional: Disentanglement loss
        disent_loss = self.disentanglement_loss(activations)
        
        return recon_loss + 0.1 * sparsity_loss + 0.1 * l1_loss + 0.1 * disent_loss
