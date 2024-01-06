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

B, s, i, j, d = 1, 512, 227, 227, 32
x = torch.randn(B, s, i, d).cuda()
model = MSAColumnGlobalAttention(d, 8).cuda()

print("x shape: ", x.shape)
print("model(x) shape: ", model(x).shape)