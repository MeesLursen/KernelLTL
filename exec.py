import numpy as np
import math
#import tensorflow as tf
from kernel_class_np import LTLKernel

import time

start = time.time()
T       = 20
AP      = 5
seed    = 1
eps     = 0.01
delta   = 1 - 0.99
N       = math.ceil((2 / eps**2) * np.log(2 / delta))
m       = 2000


kernel = LTLKernel(T, AP, seed)
kernel.sample_traces_kernel(N)
kernel.sample_formulas_kernel(m)
kernel.build_F()
print(kernel.F.shape)
print(np.unique(kernel.F))
kernel.build_K()
print(kernel.K.shape)
print(kernel.K)

print("K diag (should equal N):", np.unique(np.round(np.diag(kernel.K))))
print("K symmetric?", np.all(np.abs(kernel.K-kernel.K.T) == 0))
print("K sample values:", kernel.K[:5,:5])



end = time.time()
print(end - start)