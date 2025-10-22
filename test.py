from formula_utils import sample_formulas
from kernel_class import LTLKernel

kernel = LTLKernel(20,5,1)
ls = kernel.sample_dataset_formulas_kernel(50, 0.45,2,True)

for phi in ls:
    print(phi)
    