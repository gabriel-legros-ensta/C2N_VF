import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder

class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim, hidden_dim_list, dropout, beta=0.05):
        super().__init__()
        # input_dim = 5000 + 4 = 5004, cond_dim = 1 (label)
        self.encoder = Encoder(input_dim + cond_dim, latent_dim, hidden_dim_list=hidden_dim_list, dropout=dropout)
        self.decoder = Decoder(latent_dim + cond_dim, input_dim, hidden_dim_list=hidden_dim_list[::-1], dropout=dropout)
        self.beta = beta
        self.cond_dim = cond_dim

    def encode(self, x, label):
        # x: (n, 5004), label: (n, 1)
        x_cond = torch.cat([x, label], dim=1)
        return self.encoder(x_cond)
    
    def decode(self, z, label):
        # z: (n, latent_dim), label: (n, 1)
        z_cond = torch.cat([z, label], dim=1)
        return self.decoder(z_cond)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, label):
        mu, log_var = self.encode(x, label)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z, label)
        return recon, mu, log_var

    def loss(self, recon, target, mu, log_var):
        recon_loss = F.mse_loss(recon, target, reduction='mean')
        kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + self.beta * kl_div
