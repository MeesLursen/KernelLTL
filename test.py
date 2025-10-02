# import torch
import numpy as np
from formula_class_torch import sample_traces_torch, eval_traces_batch_torch, Globally, Not, Atom, Eventually, Next

# device = 'cpu'
# rng = torch.Generator(device = 'cpu').manual_seed(1)

trace = np.zeros((5,20), dtype=np.bool)

trace[:3,0] = np.array([True])
print(trace)

#out = np.where(trace[:, 0], np.array([1], dtype = np.int8), np.array([-1], dtype = np.int8))
out = np.where(trace[:, 0], 1, -1)

print(out.shape)
# traces = sample_traces_torch(512,5,20,rng,device)


# Atom_0 = Atom(('p2',2))
# Atom_2 = Atom(('p4',4))

# formula_0 = Globally(Not(Atom(('p2',2))))
# formula_2 = Eventually(Next(Not(Atom(('p4',4)))))


# print(formula_0)
# print(formula_2)

# sats_0 = eval_traces_batch_torch(formula_0, traces)
# sats_2 = eval_traces_batch_torch(formula_2, traces)

# vals_0 = sats_0[:,0]
# vals_2 = sats_2[:,0]
# print(vals_0)
# print(vals_2)
