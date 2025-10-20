from kernel_class import LTLKernel

T = 20 
AP = 5
seed = 1


kernel = LTLKernel(T, AP, seed)

kernel.construct_anchor_formulas_kernel(1024)
print(kernel.anchor_formulas)
print(len(kernel.anchor_formulas))