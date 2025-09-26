import numpy as np
import math
#import tensorflow as tf
from kernel_class import LTLKernel
from formula_class import random_formula, Formula, Atom, Not, And
import time

start = time.time()
len_tr  = 20
n_ap    = 5
eps     = 0.01
delta   = 1 - 0.99
N       = math.ceil((2 / eps**2) * np.log(2 / delta))


kernel = LTLKernel(len_tr, n_ap)
# sampled_tr = kernel.sample_traces_1(N, seed = 1)
# sampled_tr = kernel.sample_traces_2(N, seed = 1)
sampled_formula = random_formula(n_atoms = n_ap)
# test_trace : np.ndarray = sampled_tr[0]


print(sampled_formula)
print(sampled_formula.atoms())

# for (_,i) in sampled_formula.atoms():
#     print(f"val for p_{i}: {test_trace[i,:]}")

# print(f"formula val: {sampled_formula.eval_trace(test_trace)}")



end = time.time()
print(end - start)