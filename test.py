import torch

x = torch.randn(1, 10)
y = torch.randn(10, 1)
z = x - y

print(z, z.shape)