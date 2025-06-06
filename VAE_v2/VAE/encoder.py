import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim_list=None, dropout=0.1):
        super().__init__()

        self.hidden_dim_list = hidden_dim_list

        # Définition des couches linéaires
        self.fc_layers = nn.ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dim_list:
            self.fc_layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim

        self.fc_mu = nn.Linear(prev_dim, latent_dim)      # μ
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)  # logσ²
        self.dropout = nn.Dropout(dropout)

    def forward(self, xy):
        h = xy
        for fc in self.fc_layers:
            h = F.relu(fc(h))
            h = self.dropout(h)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var
