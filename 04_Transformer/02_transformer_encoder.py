import torch
import torch.nn as nn

trans = nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    batch_first=True,
    norm_first=True
)

encoder = nn.TransformerEncoder(trans, num_layers=6)

print(encoder)

x = torch.randint(0, 100, (1, 10))

src_embed = nn.Embedding(100, 512)

src_emb = src_embed(x)

print(src_emb.shape)

out = encoder(src_emb)
print(out.shape)