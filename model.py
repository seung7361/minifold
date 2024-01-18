import torch
from typing import Tuple

from Rigid import Rotation, Rigid

import math

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
    def __init__(self, c=32, c_z=32, n_head=4):
        super().__init__()
        self.c = c
        self.c_z = c_z
        self.n_head = n_head

        self.layer_norm = torch.nn.LayerNorm(c)

        self.query = torch.nn.Linear(c, c * n_head, bias=False)
        self.key = torch.nn.Linear(c, c * n_head, bias=False)
        self.value = torch.nn.Linear(c, c * n_head, bias=False)

        self.bias = torch.nn.Linear(c, n_head, bias=False)
        self.gate = torch.nn.Linear(c, c * n_head)
        self.output = torch.nn.Linear(c * n_head, c)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, z):
        """
        Algorithm 13: Triangular gated self-attention around starting node

        z: (B, i, j, c)

        return: (B, i, j, c_z)
        """
        
        z = self.layer_norm(z) # (B, i, j, c)

        q, k, v = self.query(z), self.key(z), self.value(z) # (B, i, j, c * h)
        b = self.bias(z) # (B, i, j, h)
        g = torch.sigmoid(self.gate(z)) # (B, i, j, c * h)

        B, i, j, _ = q.shape
        h, c = self.n_head, self.c

        q = q.view(B, i, j, h, c) # (B, i, j, h, c)
        k = k.view(B, i, j, h, c)
        v = v.view(B, i, j, h, c)
        g = g.view(B, i, j, h, c)
        b = b.view(B, i, j, 1, self.n_head) # (B, i, j, 1, h)

        a = torch.einsum("b i q h c, b i v h c -> b i q v h", q, k) * (self.c ** -0.5) + b # (B, i, r_q, r_v, h)
        a = torch.nn.functional.softmax(a, dim=-2) # (B, i, r_q, r_v, h)
        
        o = g * torch.einsum("b i q v h, b i v h c -> b i q h c", a, v) # (B, i, r_q, h, c)
        o = o.view(B, i, j, self.n_head * self.c) # (B, i, j, h * c)

        return self.output(o) # (B, i, j, c_z)
    

class TriangleAttentionEndingNode(torch.nn.Module):
    def __init__(self, c=32, c_z=32, n_head=4):
        super().__init__()
        self.c = c
        self.c_z = c_z
        self.n_head = n_head

        self.layer_norm = torch.nn.LayerNorm(c)

        self.query = torch.nn.Linear(c, c * n_head, bias=False)
        self.key = torch.nn.Linear(c, c * n_head, bias=False)
        self.value = torch.nn.Linear(c, c * n_head, bias=False)

        self.bias = torch.nn.Linear(c, n_head, bias=False)
        self.gate = torch.nn.Linear(c, c * n_head)
        self.output = torch.nn.Linear(c * n_head, c)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, z):
        """
        Algorithm 14: Triangular gated self-attention around ending node

        z: (B, i, j, c)

        return: (B, i, j, c)
        """
        
        z = z.transpose(-2, -3) # Ending node, flips i and j
        z = self.layer_norm(z) # (B, i, j, c)

        q, k, v = self.query(z), self.key(z), self.value(z) # (B, i, j, c * h)
        b = self.bias(z) # (B, i, j, h)
        g = torch.sigmoid(self.gate(z)) # (B, i, j, c * h)

        B, i, j, _ = q.shape
        h, c = self.n_head, self.c

        q = q.view(B, i, j, h, c) # (B, i, j, h, c)
        k = k.view(B, i, j, h, c)
        v = v.view(B, i, j, h, c)
        g = g.view(B, i, j, h, c)
        b = b.view(B, i, j, 1, self.n_head) # (B, i, j, 1, h)

        a = torch.einsum("b i q h c, b i v h c -> b i q v h", q, k) * (self.c ** -0.5) + b # (B, i, r_q, r_v, h)
        a = torch.nn.functional.softmax(a, dim=-2)

        o = g * torch.einsum("b i q v h, b i v h c -> b i q h c", a, v) # (B, i, r_q, h, c)
        o = o.view(B, i, j, self.n_head * self.c) # (B, i, j, h * c)

        return self.output(o).transpose(-2, -3) # (B, i, j, c_z)


class TriangleMultiplicationOutgoing(torch.nn.Module):
    def __init__(self, c=128):
        super().__init__()

        self.c = c

        self.layer_norm = torch.nn.LayerNorm(c)
        self.layer_norm_out = torch.nn.LayerNorm(c)

        self.proj_a = torch.nn.Linear(c, c)
        self.gate_a = torch.nn.Linear(c, c)
        self.proj_b = torch.nn.Linear(c, c)
        self.gate_b = torch.nn.Linear(c, c)
        self.gate = torch.nn.Linear(c, c)
        self.proj_o = torch.nn.Linear(c, c)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, z):
        """
        Algorithm 11: Triangular multiplicative update using "outgoing" edges

        z: (B, i, j, c)

        return: (B, i, j, c)
        """

        z = self.layer_norm(z)

        a = self.proj_a(z) * self.sigmoid(self.gate_a(z))
        b = self.proj_b(z) * self.sigmoid(self.gate_b(z))
        gate = self.sigmoid(self.gate(z))

        return gate * self.proj_o(self.layer_norm_out(torch.einsum("... i k c, ... j k c -> ... i j c", a, b)))


class TriangleMultiplicationIncoming(torch.nn.Module):
    def __init__(self, c=128):
        super().__init__()

        self.c = c

        self.layer_norm = torch.nn.LayerNorm(c)
        self.layer_norm_out = torch.nn.LayerNorm(c)

        self.proj_a = torch.nn.Linear(c, c)
        self.gate_a = torch.nn.Linear(c, c)
        self.proj_b = torch.nn.Linear(c, c)
        self.gate_b = torch.nn.Linear(c, c)
        self.gate = torch.nn.Linear(c, c)
        self.proj_o = torch.nn.Linear(c, c)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, z):
        """
        Algorithm 12: Triangular multiplicative update using "incoming" edges

        z: (B, i, j, c)

        return: (B, i, j, c)
        """

        z = self.layer_norm(z)

        a = self.proj_a(z) * self.sigmoid(self.gate_a(z))
        b = self.proj_b(z) * self.sigmoid(self.gate_b(z))
        gate = self.sigmoid(self.gate(z))

        return gate * self.proj_o(self.layer_norm_out(torch.einsum("... k i c, ... k j c -> ... i j c", a, b)))


class PairTransition(torch.nn.Module):
    def __init__(self, c=128, n=4):
        super().__init__()

        self.c = c
        self.n = n

        self.layer_norm = torch.nn.LayerNorm(c)
        self.proj_in = torch.nn.Linear(c, c * 4)
        self.proj_out = torch.nn.Linear(c * 4, c)
        self.relu = torch.nn.ReLU()

    def forward(self, z):
        """
        Algorithm 15: Transition layer in the pair stack

        z: (B, i, j, c)

        return: (B, i, j, c)
        """

        z = self.layer_norm(z)

        z = self.proj_in(z)
        z = self.relu(z)
        z = self.proj_out(z)

        return z


class TemplatePairStackBlock(torch.nn.Module):
    def __init__(self, c=64, n=4, n_head=4, p=0.25):
        super().__init__()

        self.c = c
        self.n = n
        self.n_head = n_head
        self.p = p

        self.dropout_row = DropoutRowwise(p)
        self.dropout_col = DropoutColumnwise(p)

        self.tri_attn_start = TriangleAttentionStartingNode(c, c, n_head)
        self.tri_attn_end = TriangleAttentionEndingNode(c, c, n_head)
        self.tri_mult_out = TriangleMultiplicationOutgoing(c)
        self.tri_mult_in = TriangleMultiplicationIncoming(c)
        self.pair_transition = PairTransition(c, n)
    
    def forward(self, t):
        """
        Algorithm 16: Template Pair Stack (Block)

        t: (B, i, j, c)

        return: (B, i, j, c)
        """

        t += self.dropout_row(self.tri_attn_start(t))
        t += self.dropout_col(self.tri_attn_end(t))
        t = self.dropout_row(self.tri_mult_out(t))
        t = self.dropout_row(self.tri_mult_in(t))
        t += self.pair_transition(t)

        return t


class TemplatePairStack(torch.nn.Module):
    def __init__(self, n_block, c=64, n=4, n_head=4, p=0.25):
        super().__init__()

        self.n_block = n_block
        self.c = c
        self.n = n
        self.n_head = n_head
        self.p = p

        self.blocks = torch.nn.ModuleList([
                TemplatePairStackBlock(c, n, n_head, p)
                for _ in range(n_block)
        ])
        self.layer_norm = torch.nn.LayerNorm(c)
    
    def forward(self, t):
        for block in self.blocks:
            t = block(t)

        return self.layer_norm(t)


class TemplatePointwiseAttention(torch.nn.Module):
    def __init__(self, c=64, n_head=4):
        super().__init__()

        self.c = c
        self.n_head = n_head

        self.proj_q = torch.nn.Linear(c, c * n_head, bias=False)
        self.proj_k = torch.nn.Linear(c, c * n_head, bias=False)
        self.proj_v = torch.nn.Linear(c, c * n_head, bias=False)

        self.proj_o = torch.nn.Linear(c * n_head, c)
    
    def forward(self, t, z):
        """
        Algorithm 17: Template Pointwise Attention

        t: (B, s, i, j, c). s: N_templ, i: N_res, j: N_res
        z: (B, i, j, c)

        return: (B, i, j, c)
        """

        q = self.proj_q(z) # (B, i, j, c * h)
        k = self.proj_k(t) # (B, s, i, j, c * h)
        v = self.proj_v(t)

        B, s, i, j, _ = k.shape
        h, c = self.n_head, self.c

        q = q.view(B, i, j, h, c) # (B, i, j, h, c)
        k = k.view(B, s, i, j, h, c) # (B, s, i, j, h, c)
        v = v.view(B, s, i, j, h, c)

        a = torch.einsum("b i j h c, b s i j h c -> b s i j h", q, k) * (self.c ** -0.5)
        a = torch.nn.functional.softmax(a, dim=1) # (B, s, i, j, h)

        o = torch.einsum("b s i j h, b s i j h c -> b i j h c", a, v) # (B, i, j, h, c)
        o = o.view(B, i, j, h * c)

        return self.proj_o(o) # (B, i, j, c)


class MSARowAttentionWithPairBias(torch.nn.Module):
    def __init__(self, c=32, n_head=8):
        super().__init__()

        self.c = c
        self.n_head = n_head

        self.layer_norm = torch.nn.LayerNorm(c)
        self.layer_norm_b = torch.nn.LayerNorm(c)

        self.proj_q = torch.nn.Linear(c, c * n_head, bias=False)
        self.proj_k = torch.nn.Linear(c, c * n_head, bias=False)
        self.proj_v = torch.nn.Linear(c, c * n_head, bias=False)
        self.proj_b = torch.nn.Linear(c, n_head, bias=False)
        self.proj_g = torch.nn.Linear(c, c * n_head, bias=False)
        self.proj_o = torch.nn.Linear(c * n_head, c)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, m, z):
        """
        Algorithm 7: MSA row-wise gated self-attention with pair bias

        m: (B, s, i, c)
        z: (B, i, j, c)

        return: (B, s, i, c)
        """

        m = self.layer_norm(m)

        q, k, v = self.proj_q(m), self.proj_k(m), self.proj_v(m)
        b = self.proj_b(self.layer_norm_b(z)) # (B, i, j, h)
        gate = self.sigmoid(self.proj_g(m))

        B, s, i, _ = q.shape
        h, c = self.n_head, self.c

        q = q.view(B, s, i, h, c) # (B, s, i, h, c)
        k = k.view(B, s, i, h, c)
        v = v.view(B, s, i, h, c)
        b = b.view(B, 1, i, j, h) # (B, 1, i, j, h)

        a = torch.einsum("b s i h c, b s j h c -> b s i j h", q, k) * (self.c ** -0.5) + b # (B, s, i, j, h)
        a = torch.nn.functional.softmax(a, dim=-2) # (B, s, i, j, h)

        o = gate * torch.einsum("b s i j h, b s j h c -> b s i h c", a, v) # (B, s, i, h, c)
        o = o.view(B, s, i, h * c) # (B, s, i, h * c)

        return self.proj_o(o) # (B, s, i, c)

    
class MSAColumnAttention(torch.nn.Module):
    def __init__(self, c=32, n_head=8):
        super().__init__()

        self.c = c
        self.n_head = n_head

        self.layer_norm = torch.nn.LayerNorm(c)

        self.proj_q = torch.nn.Linear(c, c * n_head, bias=False)
        self.proj_k = torch.nn.Linear(c, c * n_head, bias=False)
        self.proj_v = torch.nn.Linear(c, c * n_head, bias=False)
        self.proj_g = torch.nn.Linear(c, c * n_head, bias=False)
        self.proj_o = torch.nn.Linear(c * n_head, c)

    def forward(self, m):
        """
        Algorithm 8: MSA column-wise gated self-attention

        m: (B, s, i, c)

        return: (B, s, i, c)
        """

        m = self.layer_norm(m)

        q, k, v = self.proj_q(m), self.proj_k(m), self.proj_v(m)
        gate = self.sigmoid(self.proj_g(m))

        B, s, i, _ = q.shape
        h, c = self.n_head, self.c

        q = q.view(B, s, i, h, c)
        k = k.view(B, s, i, h, c)
        v = v.view(B, s, i, h, c)

        a = torch.einsum("b s i h c, b t i h c -> b s t i h", q, k) * (self.c ** -0.5) # (B, s, t, i, h)
        a = torch.nn.functional.softmax(a, dim=-3)

        o = gate * torch.einsum("b s t i h, b s t h c -> b s i h c", a, v) # (B, s, i, h, c)
        o = o.view(B, s, i, h * c)

        return self.proj_o(o) # (B, s, i, c)


class MSAColumnGlobalAttention(torch.nn.Module):
    def __init__(self, c=8, n_head=8):
        super().__init__()

        self.c = c
        self.n_head = n_head

        self.layer_norm = torch.nn.LayerNorm(c)
        self.proj_q = torch.nn.Linear(c, c * n_head, bias=False)
        self.proj_k = torch.nn.Linear(c, c, bias=False)
        self.proj_v = torch.nn.Linear(c, c, bias=False)
        self.proj_g = torch.nn.Linear(c, c * n_head, bias=False)
        self.proj_o = torch.nn.Linear(c * n_head, c)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, m):
        """
        Algorithm 19: MSA global column-wise gated self-attention

        m: (B, s, i, c)

        return: (B, s, i, c)
        """

        m = self.layer_norm(m)

        q, k, v = self.proj_q(m), self.proj_k(m), self.proj_v(m) # (B, s, i, c * h), (B, s, i, c), (B, s, i, c)
        B, s, i, _ = q.shape
        h, c = self.n_head, self.c
        q = q.view(B, s, i, h, c).mean(dim=1) # (B, i, h, c)
        gate = self.sigmoid(self.proj_g(m)).view(B, s, i, h, c) # (B, s, i, h, c)

        a = torch.einsum("b i h c, b t i c -> b t i h", q, k) * (self.c ** -0.5) # (B, t, i, h)
        a = torch.nn.functional.softmax(a, dim=-3)

        o = gate * torch.einsum("b t i h, b t i c -> b t i h c", a, v) # (B, t, i, h, c)
        o = o.view(B, s, i, h * c)

        return self.proj_o(o) # (B, s, i, c)


class MSATransition(torch.nn.Module):
    def __init__(self, c=32, n=4):
        super().__init__()

        self.c = c
        self.n = n

        self.layer_norm = torch.nn.LayerNorm(c)
        self.proj_in = torch.nn.Linear(c, c * 4)
        self.proj_out = torch.nn.Linear(c * 4, c)
        self.relu = torch.nn.ReLU()

    def forward(self, m):
        """
        Algorithm 9: Transition layer in the MSA Stack
        
        m: (B, s, i, c)

        return: (B, s, i, c)
        """

        m = self.layer_norm(m)

        m = self.proj_in(m)
        m = self.relu(m)
        m = self.proj_out(m)

        return m


class OuterProductMean(torch.nn.Module):
    def __init__(self, c=32):
        super().__init__()

        self.c = c

        self.layer_norm = torch.nn.LayerNorm(c)
        self.linear_a = torch.nn.Linear(c, c)
        self.linear_b = torch.nn.Linear(c, c)
        self.linear_out = torch.nn.Linear(c * c, c)

    def forward(self, m):
        """
        Algorithm 10: Outer product mean

        m: (B, s, i, c)

        return: (B, i, j, c)
        """

        m = self.layer_norm(m)

        a, b = self.linear_a(m), self.linear_b(m)
        outer = a.unsqueeze(2) * b.unsqueeze(3)
        outer = outer.mean(dim=1)

        return self.linear_out(outer) # (B, i, j, c)


class ExtraMSAStackBlock(torch.nn.Module):
    def __init__(self, c=8, n=4, n_head=4, p=0.25):
        super().__init__()

        self.c = c
        self.n = n
        self.n_head = n_head
        self.p = p

        self.dropout_row = DropoutRowwise(p)
        self.dropout_col = DropoutColumnwise(p)

        self.msa_row_attn = MSARowAttentionWithPairBias(c, n_head)
        self.msa_col_attn = MSAColumnGlobalAttention(c, n_head)
        self.msa_col_global_attn = MSAColumnGlobalAttention(c, n_head)
        self.msa_transition = MSATransition(c, n)
        self.outer_product_mean = OuterProductMean(c)

        self.tri_mul_out = TriangleMultiplicationOutgoing(c)
        self.tri_mul_in = TriangleMultiplicationIncoming(c)
        self.tri_attn_start = TriangleAttentionStartingNode(c, c, n_head)
        self.tri_attn_end = TriangleAttentionEndingNode(c, c, n_head)
        self.pair_transition = PairTransition(c, n)
    
    def forward(self, e, z):
        """
        Algorithm 18: Extra MSA Stack (Block)

        e: (B, s, i, c)
        z: (B, i, j, c)

        return: (B, i, j, c)
        """

        # MSA Stack
        e += self.dropout_row(self.msa_row_attn(e, z))
        e += self.msa_col_attn(e)
        e += self.msa_transition(e)

        # Communication
        z += self.outer_product_mean(e)

        # Pair Stack
        z += self.dropout_row(self.tri_mul_out(z))
        z += self.dropout_row(self.tri_mul_in(z))
        z += self.dropout_row(self.tri_attn_start(z))
        z += self.dropout_col(self.tri_attn_end(z))
        z += self.pair_transition(z)

        return z
    

class ExtraMSAStack(torch.nn.Module):
    def __init__(self, n_block, c=8, n=4, n_head=4, p=0.25):
        super().__init__()

        self.n_block = n_block
        self.c = c
        self.n = n
        self.n_head = n_head
        self.p = p

        self.blocks = torch.nn.ModuleList([
                ExtraMSAStackBlock(c, n, n_head, p)
                for _ in range(n_block)
        ])
    
    def forward(self, e, z):
        for block in self.blocks:
            e = block(e, z)

        return e


class EvoformerBlock(torch.nn.Module):
    def __init__(self, n_block=48, c_s=384, n_head=8, p=0.25):
        super().__init__()

        self.n_block = n_block
        self.c_s = c_s

        self.dropout_row = DropoutRowwise(p)
        self.dropout_col = DropoutColumnwise(p)

        self.msa_row_attn = MSARowAttentionWithPairBias(c_s, n_head)
        self.msa_col_attn = MSAColumnAttention(c_s, n_head)
        self.msa_transition = MSATransition(c_s)

        self.outer_product_mean = OuterProductMean(c_s)

        self.tri_mul_out = TriangleMultiplicationOutgoing(c_s)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_s)
        self.tri_attn_start = TriangleAttentionStartingNode(c_s, c_s, n_head)
        self.tri_attn_end = TriangleAttentionEndingNode(c_s, c_s, n_head)
        self.pair_transition = PairTransition(c_s)

    def forward(self, m, z):
        """
        Algorithm 6: Evoformer stack (Block)

        m: (B, s, i, c)
        z: (B, i, j, c)

        return: [
            m: (B, s, i, c),
            z: (B, i, j, c)
        ]
        """

        # MSA Stack
        m += self.dropout_row(self.msa_row_attn(m, z))
        m += self.msa_col_attn(m)
        m += self.msa_transition(m)

        # Communication
        z += self.outer_product_mean(m)

        # Pair Stack
        z += self.dropout_row(self.tri_mul_out(z))
        z += self.dropout_row(self.tri_mul_in(z))
        z += self.dropout_row(self.tri_attn_start(z))
        z += self.dropout_col(self.tri_attn_end(z))
        z += self.pair_transition(z)

        return m, z


class Evoformer(torch.nn.Module):
    def __init__(self, n_block=48, c_s=384, n_head=8, p=0.25):
        super().__init__()

        self.n_block = n_block
        self.c_s = c_s
        self.n_head = n_head
        self.p = p

        self.blocks = torch.nn.ModuleList([
                EvoformerBlock(n_block, c_s, n_head, p)
                for _ in range(n_block)
        ])
        self.proj_o = torch.nn.Linear(c_s, c_s)
    
    def forward(self, m, z):
        """
        return: [
            m: (B, s, i, c),
            z: (B, i, j, c),
            s: (B, i, c_s)
        ]
        """
        for block in self.blocks:
            m, z = block(m, z)

        s = self.proj_o(m[:, 0, :, :])

        return m, z, s


class InvariantPointAttention(torch.nn.Module):
    def __init__(self, c=16, n_head=12, q_points=4, v_points=8):
        super().__init__()

        self.c = c
        self.n_head = n_head
        self.q_points = q_points
        self.v_points = v_points

        self.query = torch.nn.Linear(c, c * n_head, bias=False)
        self.key = torch.nn.Linear(c, c * n_head, bias=False)
        self.value = torch.nn.Linear(c, c * n_head, bias=False)
        self.bias = torch.nn.Linear(c, n_head, bias=False)

        self.query_points = torch.nn.Linear(c, 3 * q_points * n_head, bias=False)
        self.key_points = torch.nn.Linear(c, 3 * q_points * n_head, bias=False)
        self.value_points = torch.nn.Linear(c, 3 * v_points * n_head, bias=False)

        w_c = torch.tensor((2 / (9 * self.q_points)) ** 0.5, requires_grad=False)
        w_L = torch.tensor((1 / 3) ** 0.5, requires_grad=False)
        self.gamma = torch.nn.Parameter(torch.zeros(n_head) * 0.541324854612918)

        self.register_buffer("w_c", w_c)
        self.register_buffer("w_L", w_L)

        self.linear_out = torch.nn.Linear(c * n_head * 3, c)
        self.softplus = torch.nn.Softplus()
    

    def forward(self, s, z, T: Rigid):
        """
        Algorithm 22: Invariant point attention (IPA)

        s: (B, i, c)
        z: (B, i, j, c)
        T: Rigid, (B, i) -> transformation object
        
        return: (B, i, c)
        """

        B, i, c = s.shape
        h = self.n_head
        q, k, v = self.query(s), self.key(s), self.value(s) # (B, i, c * h)
        q, k, v = q.view(B, i, h, c), k.view(B, i, h, c), v.view(B, i, h, c)

        q_points, k_points, v_points = self.query_points(s), self.key_points(s), self.value_points(s)
        q_points, k_points, v_points = q_points.view(B, i, h * self.q_points, 3), k_points.view(B, i, h * self.q_points, 3), v_points.view(B, i, h * self.v_points, 3)
        # (B, i, h * q_points, 3), (B, i, h * k_points, 3), (B, i, h * v_points, 3)

        # (B, i, h, q_points, 3)
        q_points = T[..., None].apply(q_points).view(B, i, h, self.q_points, 3)
        k_points = T[..., None].apply(k_points).view(B, i, h, self.q_points, 3)
        v_points = T[..., None].apply(v_points).view(B, i, h, self.v_points, 3)

        b = self.bias(z).permute(0, 3, 1, 2) # (B, i, j, h)
        a = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1) * (self.c ** -0.5) + b # (B, h, i, j)


        point_attention = torch.sum((q_points.unsqueeze(-4) - k_points.unsqueeze(-5)) ** 2, dim=-1) # (B, i, j, h, q_points)
        head_weights = self.softplus(self.gamma).view(1, 1, 1, h) # (1, 1, 1, h, 1)
        point_attention = torch.sum(point_attention, dim=-1) # (B, i, j, h)
        point_attention *= head_weights * self.w_c * (-0.5)

        a += point_attention.permute(0, 3, 1, 2) # (B, h, i, j)
        a *= self.w_L
        a = torch.nn.functional.softmax(a, dim=-2) # (B, h, i, j)

        o = torch.einsum("b h i j, b i h c -> b i h c", a, v) # (B, i, h, c)
        o = o.view(B, i, h * c) # (B, i, h * c)

        v_points = v_points.permute(0, 2, 4, 1, 3) # (B, h, 3, i, v_points)
        o_points = torch.sum(a[..., None, :, :, None] * v_points[..., None, :, :], dim=-2)
        # (B, h, 3, i, v_points)
        o_points = o_points.permute(0, 3, 1, 4, 2) # (B, i, h, v_points, 3)

        o_points_norm = torch.norm(o_points, dim=-1, keepdim=False) # (B, i, h, v_points)
        o_points_norm = torch.clamp(o_points_norm, min=1e-6).view(B, i, h * self.v_points) # (B, i, h * v_points)
        o_points = o_points.reshape(B, i, h * self.v_points * 3) # (B, i, h * v_points, 3)

        o_pair = a.transpose(-2, -3) @ z # (B, i, h, j) @ (B, i, j, c) -> (B, i, h, c)
        o_pair = o_pair.view(B, i, h * c) # (B, i, h * c)

        s = self.linear_out(torch.cat([o, o_pair, o_points, o_points_norm], dim=-1)) # (B, i, c)

        return s


class BackboneUpdate(torch.nn.Module):
    def __init__(self, c):
        super().__init__()

        self.c = c
        self.proj_b = torch.nn.Linear(c, 1)
        self.proj_c = torch.nn.Linear(c, 1)
        self.proj_d = torch.nn.Linear(c, 1)
        self.proj_t = torch.nn.Linear(c, 3)

    
    def forward(self, s):
        """
        Algorithm 23: Backbone update

        s: (B, i, c)

        b, c, d: (B, i, 1)
        t: (B, i, 3)

        return: 
        """

        b, c, d = self.proj_b(s), self.proj_c(s), self.proj_d(s)
        t = self.proj_t(s)

        quats = torch.cat([torch.ones_like(b), b, c, d], dim=-1)
        quats = quats / torch.norm(quats, dim=-1, keepdim=True)

        rot = Rotation(quats=quats)
        R = rot.rot_mats

        rigid = Rigid(R, t)

        return rigid


def torsion_angle_loss(a, a_true, a_alt):
    """
    Algorithm 27: Side chain and backbone torsion angle loss

    a: (B, i, 7, 2)
    a_true: (B, i, 7, 2)
    a_alt: (B, i, 7, 2)

    return: loss tensor
    """

    l = torch.norm(a, dim=-1, keepdim=True) # (B, i, 7, 1)
    a = a / l # (B, i, 7, 2)

    left = torch.norm(a - a_true, dim=-1)
    right = torch.norm(a - a_alt, dim=-1)
    L_torsion = torch.mean(torch.minimum(left ** 2, right ** 2), dim=(-1, -2)) # (B,)
    L_anglenorm = torch.mean(torch.abs(l - 1), dim=(-1, -2)) # (B,)

    return L_torsion + 0.02 * L_anglenorm
       

def compute_fape(T, T_true, x, x_true, z=10, d_clamp=10, eps=1e-4):
    """
    Algorithm 28: Compute the Frame aligned point error

    T: Rigid, (B, i) -> prediction
    T_true: Rigid, (B, i) -> ground truth
    both T with 3x3 rotation matrix

    x: (B, i, 3) -> prediction
    x_true: (B, i, 3) -> ground truth

    return: loss tensor
    """

    x = T.invert()[..., None].apply(x[..., None, :, :])
    x_true = T_true.invert()[..., None].apply(x_true[..., None, :, :])

    d = torch.sqrt(torch.norm(x - x_true, dim=-1) + eps)
    d = torch.clamp(d, min=0, max=d_clamp)

    L_fape = torch.mean(d, dim=-1) * (1 / z)

    return L_fape


class StructureModuleTransitionLayer(torch.nn.Module):
    def __init__(self, c):
        super().__init__()

        self.c = c

        self.linear_1 = torch.nn.Linear(c, c)
        self.linear_2 = torch.nn.Linear(c, c)
        self.linear_3 = torch.nn.Linear(c, c)

        self.relu = torch.nn.ReLU()
    
    def forward(self, s):
        s = s + self.linear_3(self.relu(self.linear_2(self.relu(self.linear_1(s)))))

        return s

class StructureModuleTransition(torch.nn.Module):
    def __init__(self, c, n_layers, p=0.1):
        super().__init__()

        self.c = c
        self.n_layers = n_layers

        self.layers = torch.nn.ModuleList([
                StructureModuleTransitionLayer(c)
                for _ in range(n_layers)
        ])
        self.dropout = torch.nn.Dropout(p)
        self.layer_norm = torch.nn.LayerNorm(c)
    
    def forward(self, s):
        for layer in self.layers:
            s = layer(s)

        return self.layer_norm(self.dropout(s))


class AngleResnetBlock(torch.nn.Module):
    def __init__(self, c=128):
        super().__init__()

        self.c = c

        self.linear_1 = torch.nn.Linear(c, c)
        self.linear_2 = torch.nn.Linear(c, c)

        self.relu = torch.nn.ReLU()

    def forward(self, a):
        a = a + self.linear_2(self.relu(self.linear_1(torch.relu(a))))

        return a


class AngleResnet(torch.nn.Module):
    def __init__(self, c=128, n_layer=8, n_angle=7):
        super().__init__()

        self.c = c
        self.n_layer = n_layer
        self.n_angle = n_angle

        self.linear_in = torch.nn.Linear(c, c)
        self.linear_initial = torch.nn.Linear(c, c)

        self.layers = torch.nn.ModuleList([
                AngleResnetBlock(c)
                for _ in range(n_layer)
        ])
        self.layer_norm = torch.nn.LayerNorm(c)
        self.linear_out = torch.nn.Linear(c, n_angle * 2)

        self.relu = torch.nn.ReLU()
    
    def forward(self, s, s_initial):
        """
        s, s_initial: (B, i, c)

        return: [
            unnomarlized_a: (B, i, 7, 2)
            a: (B, i, 7, 2)
        ]
        """
        a = self.linear_in(self.relu(s)) + self.linear_initial(self.relu(s_initial))

        for layer in self.layers:
            a = a + layer(a)

        a = self.linear_out(self.relu(a)) # (B, i, 7, 2)
        a = a.view(B, i, self.n_angle, 2)

        unnomarlized_a = a.clone()
        a = a / torch.norm(a, dim=-1, keepdim=True)

        return unnomarlized_a, a


def torsion_angles_to_frames(T, angles, aatype):
    """
    Algorithm 24: Compute all atom coordinates

    T: Rigid, (B, i) -> transformation object
    angles: (B, i, 7, 2)
    aatype: (B, i) -> amino acid indices
    """

    assert angles.shape[-2] == 7
    assert angles.shape[-1] == 2

    B, n = angles.shape
    m = torch.zeros([B, n, 8, 4, 4], device=angles.device, dtype=angles.dtype)
    default_frames = Rigid.from_tensor_4x4(m)

    sin_angles = angles[..., 0]
    cos_angles = angles[..., 1]

    sin_angles = torch.cat([torch.zeros([n, 1]), sin_angles], dim=-1)
    cos_angles = torch.cat([torch.ones([n, 1]), cos_angles], dim=-1)
    zeros = torch.zeros_like(sin_angles)
    ones = torch.ones_like(sin_angles)


    """
    Algorithm 25: Make a transformation that rotates around the x-axis

    makeRotX
    """
    all_rots = Rigid.from_rot_mats(torch.stack([
        ones, zeros, zeros,
        zeros, cos_angles, -sin_angles,
        zeros, sin_angles, cos_angles
    ], dim=-1).view(B, n, 8, 3, 3))
    
    all_frames = default_frames * all_rots

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_backbone = all_frames[..., 4]
    chi2_frame_to_backbone = chi1_frame_to_backbone * chi2_frame_to_frame
    chi3_frame_to_backbone = chi2_frame_to_backbone * chi3_frame_to_frame
    chi4_frame_to_backbone = chi3_frame_to_backbone * chi4_frame_to_frame

    all_frames_to_bb = Rigid.cat([
        all_frames[..., :5],
        chi2_frame_to_backbone[..., None],
        chi3_frame_to_backbone[..., None],
        chi4_frame_to_backbone[..., None],
    ], dim=-1)

    all_frames_to_global = T[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(aatype, T):
    """
    Algorithm 24, line 11

    aatype: (B, i) -> amino acid indices
    T: Rigid, (B, i) -> transformation object
    """

    default_4x4 = torch.zeros([B, i, 14, 4, 4], device=aatype.device, dtype=aatype.dtype)
    
    t_atoms_to_global = T[..., None, :]
    t_atoms_to_global = torch.sum(t_atoms_to_global, dim=-1)

    lit_positions = torch.zeros([B, i, 14, 3], device=aatype.device, dtype=aatype.dtype)
    pred_positions = t_atoms_to_global.apply(lit_positions[..., None, :, :])

    return pred_positions


class StructureModule(torch.nn.Module):
    def __init__(self, c=128, n_layer=8, n_head=8, p=0.25):
        super().__init__()

        self.c = c
        self.n_layer = n_layer

        self.layernorm_s = torch.nn.LayerNorm(c)
        self.layernorm_z = torch.nn.LayerNorm(c)

        self.linear_in = torch.nn.Linear(c, c)
        self.ipa = InvariantPointAttention(c, n_head)
        self.dropout = torch.nn.Dropout(p)
        self.layernorm_ipa = torch.nn.LayerNorm(c)

        self.transition = StructureModuleTransition(c, n_layer, p)
        self.backbone_update = BackboneUpdate(c)
        self.angle_resnet = AngleResnet(c, n_layer)


    def forward(self, s_initial, z, aatype):
        """
        s_initial: (B, i, c)
        z: (B, i, j, c)
        aatype: (B, i) -> amino acid indices

        return: [
            s: (B, i, c),
            T: Rigid, (B, i) -> transformation object
            a: (B, i, 7, 2)
        ]
        """

        s_initial = self.layernorm_s(s_initial)
        z = self.layernorm_z(z)

        s = self.linear_in(s_initial)
        rigids = Rigid.identity(
            shape=s.shape[:-1],
            device=s.device,
            dtype=s.dtype
        )

        outputs = []
        for _ in range(self.n_layer):
            s = s + self.ipa(s, z, rigids)
            s = self.layernorm_ipa(self.dropout(s))
            s = self.transition(s)

            rigids = rigids.compose_q_update_vec(self.backbone_update(s))

            backbone_to_global = Rigid(
                Rotation(
                    rot_mats=rigids.get_rots().get_rot_mats(), 
                    quats=None
                ),
                rigids.get_trans(),
            )

            scaled_rigids = rigids.scale_translation(10)
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)

            all_frames_to_global = torsion_angles_to_frames(
                backbone_to_global, angles, aatype
            )

            pred_xyz = frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global, aatype
            )
            scaled_rigids = rigids.scale_translation(10)

            preds = {
                "frames": scaled_rigids.to_tensor_7(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
                "states": s,
            }

            outputs.append(preds)

            rigids = rigids.stop_rot_gradients()
        
        return outputs


class PerResidueLDDTCaPredictor(torch.nn.Module):
    def __init__(self, no_bins, c):
        self.no_bins = no_bins
        self.c = c

        self.layer_norm = torch.nn.LayerNorm(c)

        self.linear_1 = torch.nn.Linear(c, c)
        self.linear_2 = torch.nn.Linear(c, c)
        self.linear_3 = torch.nn.Linear(c, c)

        self.relu = torch.nn.ReLU()
    
    def forward(self, s):
        """
        Algorithm 29: Predict model confidence pLDDT

        s: (B, i, c)
        """
        s = self.layer_norm(s)

        return self.linear_3(self.relu(self.linear_2(self.relu(self.linear_1(s)))))


class DistogramHead(torch.nn.Module):
    def __init__(self, c, no_bins):
        super().__init__()

        self.no_bins = no_bins
        self.c = c

        self.linear = torch.nn.Linear(c, no_bins)
    
    def forward(self, z):
        """
        Section 1.9.8: Distogram prediction

        z: (B, i, j, c)

        return: (B, i, j, no_bins) -> distogram probability distribution
        """

        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)

        return logits # (B, i, j, no_bins)


class TMScoreHead(torch.nn.Module):
    def __init__(self, c, no_bins):
        super().__init__()

        self.no_bins = no_bins
        self.c = c

        self.linear = torch.nn.Linear(c, no_bins)

    def forward(self, z):
        """
        Section 1.9.7: TM-score prediction

        z: (B, i, j, c)

        return: (B, i, j, no_bins) -> TM-score prediction
        """

        logits = self.linear(z)
        return logits


class MaskedMSAHead(torch.nn.Module):
    def __init__(self, c):
        super().__init__()

        self.c = c

        self.linear = torch.nn.Linear(c, c)
    
    def forward(self, m):
        """
        Section 1.9.9: Masked MSA prediction

        m: (B, s, i, c) MSA Embedding

        return: (B, s, i, c) -> MSA embedding
        """

        logits = self.linear(m)
        return logits


class ExperimentallyResolvedHead(torch.nn.Module):
    def __init__(self, c):
        super().__init__()

        self.c = c
        
        self.linear = torch.nn.Linear(c, c)

    def forward(self, s):
        """
        Section 1.9.10: "Experimentally resolved" prediction

        s: (B, i, c)

        return: (B, i, c) logits
        """

        logits = self.linear(s)
        return logits


B, s, i, j, c = 1, 512, 227, 227, 32

s = torch.randn(B, i, c)
z = torch.randn(B, i, j, c)
T = Rigid(
    Rotation(rot_mats=torch.randn(B, i, 3, 3)),
    torch.randn(B, i, 3),
)

model = InvariantPointAttention(c, 8, 4, 8)
model(s, z, T)