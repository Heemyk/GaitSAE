import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SparseVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, sparsity_param=0.05, beta=1.0):
        super().__init__()

        self.input_dim = input_dim
        print(input_dim)
        self.latent_dim = latent_dim
        print(latent_dim)

        # Initialize encoder weight
        self.encoder_weight = nn.Parameter(torch.randn(self.latent_dim, self.input_dim) / np.sqrt(self.input_dim))
        self.encoder_bias = nn.Parameter(torch.zeros(self.latent_dim))
        self.decoder_bias = nn.Parameter(torch.zeros(self.input_dim))

        self.sparsity_param = sparsity_param
        self.beta = beta

    def encode(self, x):
        # Encoder with tied weights
        h = F.linear(x, self.encoder_weight, self.encoder_bias)
        mu = h  # Use the full output for mu
        log_var = torch.zeros_like(mu)  # Initialize log_var as zeros for now
        return mu, log_var

    def reparameterize(self, mu, log_var):
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Decoder with tied weights (transpose of encoder weights)
        return F.linear(z, self.encoder_weight.t(), self.decoder_bias)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var, z

    def analyze(self, x):
        """
        A method for analysis purposes that returns all outputs: x_recon, mu, log_var, and z.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var, z

    def loss_function(self, x, x_recon, mu, log_var, z):
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Sparsity loss (L1 regularization on latent activations)
        sparsity_loss = torch.sum(torch.abs(z))

        return recon_loss + self.beta * kl_loss + 0.1 * sparsity_loss