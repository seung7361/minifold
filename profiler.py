import torch
import pickle
from torch.profiler import profile, record_function, ProfilerActivity

# with open("outputs.pkl", "rb") as f:
#     outputs = pickle.load(f)
# 
# for key in outputs.keys():
#     if type(outputs[key]) == torch.Tensor:
#         print(key, outputs[key].shape)
#     elif type(outputs[key]) == list:
#         print(key, len(outputs[key]))
#     else:
#         print(key, type(outputs[key]))

from model import Alphafold2

model = Alphafold2()

B, i, c, t, s = 1, 128, 384, 1, 1
x = torch.randn(B, i, 3)
batch = {
    "aatype": torch.randint(0, 20, (B, i)),
    "residue_index": torch.randint(0, 100, (B, i)),
    "target_feat": torch.randn(B, i, c),
    "msa": torch.randn(B, s, i, c),
    "template_aatype": torch.randint(0, 20, (B, t, i)),
    "template_all_atom_positions": torch.randn(B, t, i, 37, 3),
    "template_pseudo_beta": torch.randn(B, t, i, 3),
    "template_torsion_angles_sin_cos": torch.randn(B, t, i, 7, 2),
    "template_alt_torsion_angles_sin_cos": torch.randn(B, t, i, 7, 2),
    "template_torsion_angles_mask": torch.randn(B, t, i, 7),
    "n_cycle": 1,
}

# with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
#     with record_function("model"):
#         model(batch)

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))
# print(prof.key_averages().table(sort_by="cpu_memory_usage"))

from pytorch_memlab import LineProfiler

with LineProfiler(model) as prof:
    model(batch)

prof.display()