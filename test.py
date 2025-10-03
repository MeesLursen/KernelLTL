import torch
from formula_class_torch import sample_traces_torch, eval_traces_batch_torch, Globally, Not, Atom, Eventually, Next
# from kernel_class_torch import LTLKernel_torch

device = 'cpu'
rng = torch.Generator(device = 'cpu').manual_seed(1)

traces = sample_traces_torch(10,5,20,rng,device)

formula = Not(Next(Atom(('p0',0))))

sats_formula = eval_traces_batch_torch(formula,traces)
vals = torch.where(sats_formula[:, 0], 
                   torch.tensor(1.0, dtype=torch.float32, device=device),
                   torch.tensor(-1.0, dtype=torch.float32, device=device)) 
vals = vals.unsqueeze(dim=1)
print(f'vals = {vals}')
print(vals.shape)

F = torch.randint(0,2,(15,10), dtype=torch.bool, generator=rng, device=device)
F = torch.where(F, 
                torch.tensor(1.0, dtype=torch.float32, device=device),
                torch.tensor(-1.0, dtype=torch.float32, device=device))
print(f'F = {F}')
print(F.shape)

emb = F @ vals

print(f'embeddins = {emb}')
print(emb.device)