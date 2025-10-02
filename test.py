import torch
from formula_class_torch import sample_traces_torch, sample_formulas_torch, eval_traces_batch_torch

device = 'cpu'
rng = torch.Generator(device = 'cpu').manual_seed(42)

F = torch.empty((1, 10), dtype=torch.int8)
traces = sample_traces_torch(10,5,20,rng,device)
formula = sample_formulas_torch(1,0.5,6,5,True,rng,device)[0]
print(formula)
sats = eval_traces_batch_torch(formula,traces)
print(sats.type())
vals = torch.where(sats[:, 0], 1, -1).to(torch.int8)

print(vals)
print(vals.shape)
F[0,:]= vals
print(F)
print(F.shape)

# print(trace_0)
# print(vals)

# trace_conv = torch.where(traces)

# sample_traces_torch(10, 5, 20, rng=rng, device=device)
