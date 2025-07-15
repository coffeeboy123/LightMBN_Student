import torch
from torch import nn

class PartTransformer(nn.Module):
    def __init__(self, dim=512, heads=4, depth=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, tokens):  # (B, 3, 512)
        return self.encoder(tokens)  # (B, 3, 512)