import torch
import torch.nn as nn

trans = nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    batch_first=True,
)

encoder = nn.TransformerEncoder(trans, num_layers=6)

print(encoder)