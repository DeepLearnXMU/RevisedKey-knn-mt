import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, dimension, ffn_scale):
        super().__init__()
        self.query_map = nn.Sequential(
            nn.Linear(dimension, dimension * ffn_scale),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dimension * ffn_scale, dimension)
        )
        self.layer_norm = nn.LayerNorm(dimension, eps=1e-6)
    
    def forward(self, hidden):
        return self.layer_norm(self.query_map(hidden))