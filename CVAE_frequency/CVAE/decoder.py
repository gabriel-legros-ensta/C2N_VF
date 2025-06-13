import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim_list=None, dropout=0.1):
        super().__init__()
        # latent_dim = latent + cond_dim (ex: latent_dim + 1)
        self.hidden_dim_list = hidden_dim_list

        self.fc_layers = nn.ModuleList()
        prev_dim = latent_dim  # latent_dim doit inclure la condition (label)
        for h_dim in hidden_dim_list:
            self.fc_layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim

        self.fc_out = nn.Linear(prev_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        # z: (n, latent_dim + cond_dim)
        h = z
        for fc in self.fc_layers:
            h = F.relu(fc(h))
            h = self.dropout(h)
        out = self.fc_out(h)
        return out
