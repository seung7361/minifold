from model import Alphafold2
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = Alphafold2()