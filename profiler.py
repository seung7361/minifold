import torch
import pickle
from torch.profiler import profile, record_function, ProfilerActivity

with open("outputs.pkl", "rb") as f:
    outputs = pickle.load(f)

for key in outputs.keys():
    if type(outputs[key]) == torch.Tensor:
        print(key, outputs[key].shape)
    elif type(outputs[key]) == list:
        print(key, len(outputs[key]))
    else:
        print(key, type(outputs[key]))