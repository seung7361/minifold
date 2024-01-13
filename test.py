import torch

B, h, i, j, c = 2, 3, 4, 5, 6

a = torch.randn(B, h, i, j)
v = torch.randn(B, i, h, c)

o = torch.einsum("b h i j, b i h c -> b i h c", a, v) # (B, i, h, c)

print(o.shape)