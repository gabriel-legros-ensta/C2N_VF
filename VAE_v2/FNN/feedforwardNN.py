import torch
import torch.nn as nn

class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_list, dropout=None):
        super(FeedforwardNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dim_list:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout is not None and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)