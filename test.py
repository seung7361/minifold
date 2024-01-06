import torch


B, s, i, j, c = 1, 512, 227, 227, 64

x = torch.randn(B, s, i, c).half()
y = torch.randn(B, s, j, c).half()

outer = x.unsqueeze(2) * y.unsqueeze(3)
outer = outer.mean(dim=1)
print(outer.shape)