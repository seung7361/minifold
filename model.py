import torch
from typing import Tuple
from einops import rearrange

class InputEmbedder(torch.nn.Module):
    def __init__(self, tf_dim, msa_dim, c_z=128, c_m=256, relpos_k=32):
        super().__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim
        self.c_z = c_z
        self.c_m = c_m
        self.relpos_k = relpos_k

        self.linear1 = torch.nn.Linear(tf_dim, c_z)
        self.linear2 = torch.nn.Linear(tf_dim, c_z)

        self.linear3 = torch.nn.Linear(msa_dim, c_m)
        self.linear4 = torch.nn.Linear(tf_dim, c_m)

        self.relpos_linear = torch.nn.Linear(2 * relpos_k + 1, c_z)

    def one_hot(self, x, v_bins):
        """
        Algorithm 5: One-hot encoding with nearest bin

        x: (B, N, N) -> (B, N, N, 1)
        v_bins: (1, 1, 2 * relpos_k + 1)

        return: (B, N, N, 2 * relpos_k + 1)
        """
        x = x.unsqueeze(-1)
        p = torch.argmin(torch.abs(x - v_bins), dim=-1)

        p = torch.nn.functional.one_hot(p, num_classes=v_bins.shape[-1]).float()

        return p

    
    def relpos(self, ri):
        """
        Algorithm 4: Relative Position Encoding

        ri: (B, N), residue_index
        d: (B, N, N)
        v_bins: (2 * relpos_k + 1) -> (1, 1, 2 * relpos_k + 1)

        return: (B, N, N, c_z)
        """
        d = ri.unsqueeze(-1) - ri.unsqueeze(-2)
        v_bins = torch.arange(-self.relpos_k, self.relpos_k + 1, device=d.device).view(1, 1, -1)
        p = self.relpos_linear(self.one_hot(d, v_bins))

        return p

    
    def forward(self, tf, ri, msa) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Algorithm 3: Embeddings for initial representations
        tf: (B, N_res, tf_dim)
        ri: (B, N_res), residue_index
        msa: (B, N_clust, N_res, msa_dim)

        a: (B, N_res, c_z)
        b: (B, N_res, c_z)

        z: (B, N_res, c_z, c_z)

        return: m: (B, N_clust, N_res, c_m), z: (B, N_res, c_z, c_z)
        """
        a, b = self.linear1(tf), self.linear2(tf)
        z = a[..., None, :] + b[..., None, :, :]
        z += self.relpos(ri)

        m = self.linear3(msa) + self.linear4(tf)

        return m, z


class RecylingEmbedder(torch.nn.Module):
    def __init__(self, c_z=128, c_m=256):
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z

        self.min_bin = 3.25
        self.max_bin = 20.75
        self.no_bins = 15

        self.linear = torch.nn.Linear(self.no_bins, c_z)
        self.layernorm_m = torch.nn.LayerNorm(c_m)
        self.layernorm_z = torch.nn.LayerNorm(c_z)

    def one_hot(self, x, v_bins):
        """
        Algorithm 5: One-hot encoding with nearest bin

        x: (B, N_res, N_res) -> (B, N_res, N_res, 1)
        v_bins: (1, 1, 2 * relpos_k + 1)

        return: (B, N, N, 2 * relpos_k + 1)
        """
        x = x.unsqueeze(-1)
        p = torch.argmin(torch.abs(x - v_bins), dim=-1)

        p = torch.nn.functional.one_hot(p, num_classes=v_bins.shape[-1]).float()

        return p

    def forward(self, m, z, x):
        """
        Algorithm 32: Embedding of Evoformer and Structure module outputs for recycling

        m: (B, N_res, c_m)
        z: (B, N_res, N_res, c_z)
        x: (B, N_res, 3) -> predicted C_beta coordinates

        d: (B, N_res, N_res)
        """

        m, z = self.layernorm_m(m), self.layernorm_z(z)

        v_bins = torch.linspace(self.min_bin, self.max_bin, self.no_bins, dtype=x.dtype, device=x.device, requires_grad=False)
        squared_bins = v_bins ** 2
        upper = torch.cat(
            [squared_bins[1:], squared_bins.new_tensor([1e9])], dim=-1
        )
        d = torch.sum(x[..., None, :] - x[..., None, :, :], dim=-1, keepdims=True) # (B, N_res, N_res, no_bins)
        d = ((d > squared_bins) * (d < upper)).type(x.dtype)
        d = self.linear(d) # (B, N_res, N_res, c_z)

        z = d + z

        return m, z


class DropoutRowwise(torch.nn.Module):
    def __init__(self, p=0.25):
        super().__init__()

        self.p = p
        self.dropout = torch.nn.Dropout(p)

    def forward(self, x):
        """
        Dropout for row-wise inputs

        x: (B, N_res, N_res, c_z)

        return: (B, N_res, N_res, c_z)
        """
        if not self.training:
            return x


        shape = list(x.shape)
        shape[-3] = 1 # Row-wise
        mask = x.new_ones(shape)
        mask = self.dropout(mask)

        return x * mask


class DropoutColumnwise(torch.nn.Module):
    def __init__(self, p=0.25):
        super().__init__()

        self.p = p
        self.dropout = torch.nn.Dropout(p)

    def forward(self, x):
        """
        Dropout for column-wise inputs

        x: (B, N_res, N_res, c_z)

        return: (B, N_res, N_res, c_z)
        """
        if not self.training:
            return x


        shape = list(x.shape)
        shape[-2] = 1
        mask = x.new_ones(shape)
        mask = self.dropout(mask)

        return x * mask


class TriangleAttentionStartingNode(torch.nn.Module):
    def __init__(self, c=32, n_head=4):
        super().__init__()

        self.c = c
        self.n_head = n_head
        self.d = c * n_head

        self.layernorm = torch.nn.LayerNorm(self.d)


        self.proj_q = torch.nn.Linear(self.d, self.d, bias=False)
        self.proj_k = torch.nn.Linear(self.d, self.d, bias=False)
        self.proj_v = torch.nn.Linear(self.d, self.d, bias=False)

        self.proj_b = torch.nn.Linear(self.d, self.d, bias=False)
        self.proj_g = torch.nn.Linear(self.d, self.d)

        self.proj_o = torch.nn.Linear(self.d, self.d)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, z):
        """
        Algorithm 13: Triangular gated self-attention around starting node

        z: (B, i, j, d)

        d = 128 => h = 4, c = 32

        return: (B, i, j, d)
        """

        z = self.layernorm(z) # (B, i, j, d)

        q, k, v = self.proj_q(z), self.proj_k(z), self.proj_v(z) # (B, i, j, d)
        b = self.proj_b(z) # (B, j, k, d)
        g = self.sigmoid(self.proj_g(z)) # (B, i, j, d)

        B, i, j, d = q.shape
        q = q.view(B, i, j, self.n_head, self.c)
        k = k.view(B, i, j, self.n_head, self.c)
        v = v.view(B, i, j, self.n_head, self.c)
        b = b.view(B, i, j, self.n_head, self.c)
        g = g.view(B, i, j, self.n_head, self.c)

        q = q.permute(0, 3, 1, 2, 4) # (B, h, i, j, c)
        k = k.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 1, 2, 4)
        b = b.permute(0, 3, 1, 2, 4)
        g = g.permute(0, 3, 1, 2, 4)

        a = torch.einsum("b h i j c, b h i k c -> b h j k c", q, k) # (B, h, j, k, c)
        a /= (self.c ** 0.5)
        a += b # (B, h, j, k, c)

        a = torch.nn.functional.softmax(a, dim=-2) # (B, h, j, k, c)

        o = g * torch.einsum("b h j k c, b h i k c -> b h i j c", a, v)
        o = o.permute(0, 2, 3, 1, 4)
        o = o.view(B, i, j, self.d)

        o = self.proj_o(o) # (B, i, j, d)

        return o

class TriangleAttentionEndingNode(torch.nn.Module):
    def __init__(self, c=32, n_head=4):
        super().__init__()

        self.c = c
        self.n_head = n_head
        self.d = c * n_head

        self.layernorm = torch.nn.LayerNorm(self.d)


        self.proj_q = torch.nn.Linear(self.d, self.d, bias=False)
        self.proj_k = torch.nn.Linear(self.d, self.d, bias=False)
        self.proj_v = torch.nn.Linear(self.d, self.d, bias=False)

        self.proj_b = torch.nn.Linear(self.d, self.d, bias=False)
        self.proj_g = torch.nn.Linear(self.d, self.d)

        self.proj_o = torch.nn.Linear(self.d, self.d)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, z):
        """
        Algorithm 14: Triangular gated self-attention around ending node

        z: (B, i, j, d)

        d = 128 => h = 4, c = 32

        return: (B, i, j, d)
        """

        z = self.layernorm(z) # (B, i, j, d)

        q, k, v = self.proj_q(z), self.proj_k(z), self.proj_v(z) # (B, i, j, d)
        b = self.proj_b(z) # (B, j, k, d)
        g = self.sigmoid(self.proj_g(z)) # (B, i, j, d)

        B, i, j, d = q.shape
        q = q.view(B, i, j, self.n_head, self.c)
        k = k.view(B, i, j, self.n_head, self.c)
        v = v.view(B, i, j, self.n_head, self.c)
        b = b.view(B, i, j, self.n_head, self.c)
        g = g.view(B, i, j, self.n_head, self.c)

        q = q.permute(0, 3, 1, 2, 4) # (B, h, i, j, c)
        k = k.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 1, 2, 4)
        b = b.permute(0, 3, 1, 2, 4)
        g = g.permute(0, 3, 1, 2, 4)

        a = torch.einsum("b h i j c, b h k j c -> b h k i c", q, k) # (B, h, k, i, c)
        a /= (self.c ** 0.5)
        a += b # (B, h, j, k, c)

        a = torch.nn.functional.softmax(a, dim=-2) # (B, h, j, k, c)

        o = g * torch.einsum("b h i j c, b h k j c -> b h i j c", a, v)
        o = o.permute(0, 2, 3, 1, 4)
        o = o.view(B, i, j, self.d)

        o = self.proj_o(o) # (B, i, j, d)

        return o


class TriangleMultiplicationOutgoing(torch.nn.Module):
    

B, i, j, d = 1, 227, 227, 128
x = torch.randn(B, i, j, d).cuda()
model = TriangleAttentionEndingNode().cuda()

print(model(x).shape)