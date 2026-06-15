import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

linear = nn.Linear(10, 5, device=device)
x = torch.randn(1, 10, device=device)
y = linear(x)
print(y.device, y)