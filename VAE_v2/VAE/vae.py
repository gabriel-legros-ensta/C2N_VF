import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim_list, dropout, beta=0.05):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim_list=hidden_dim_list, dropout=dropout)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dim_list=hidden_dim_list[::-1], dropout=dropout)
        self.beta = beta

    def encode(self, xy):
        return self.encoder(xy)
    
    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, xy):
        mu, log_var = self.encoder(xy)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    def loss(self, recon, target, mu, log_var):
        recon_loss = F.mse_loss(recon, target, reduction='mean')
        kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + self.beta * kl_div
