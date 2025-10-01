import torch
from formula_class_torch import sample_traces_torch

device = 'cpu'
rng = torch.Generator(device = 'cpu').manual_seed(42)

# randint = torch.randint(0, 5, (), generator=rng, device=device).item()
# print(randint)

traces = torch.randint(0,2, size=(10, 5, 20), generator=rng, dtype=torch.uint8, device=device)
trace_at = traces[:, 0]
trace_0 = trace_at[:, 0]
vals = torch.where(trace_0, 1, -1).to(torch.int8)

print(trace_0)
print(vals)

# trace_conv = torch.where(traces)

# sample_traces_torch(10, 5, 20, rng=rng, device=device)
