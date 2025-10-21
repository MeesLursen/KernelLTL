from kernel_class import LTLKernel
import math
import torch

T       = 20
AP      = 5
seed    = 1
m       = 1024

eps     = 0.01
delta   = 1 - 0.99
# N       = math.ceil((2 / eps**2) * math.log(2 * m / delta))
N = 5000



# Initialize kernel for semantic embeddings
kernel = LTLKernel(T, AP, seed)  # adjust T and AP as needed

formula_list = kernel.sample_dataset_formulas_kernel(1000,0.45,4, force_tree=True)

formula_set = set(formula_list)
formula_list_dedupe = list(formula_set)

print(len(formula_list))
print(len(formula_list_dedupe))

# print(kernel.device)
# kernel.sample_traces_kernel(N)  # adjust N based on your needs
# kernel.construct_anchor_formulas_kernel(m)  # m should match model's n_embd
# kernel.build_F()

# torch.set_printoptions(precision=None, threshold=10000, edgeitems=None, linewidth=None, profile=None, sci_mode=None)

# for i in range(N):
#     print(torch.unique(kernel.F[:, i], return_counts=True))
# print("Finished building Kernel.")