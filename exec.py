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
sampled_tr = kernel.sample_traces_1(N, seed = 1)
#sampled_tr = kernel.sample_traces_2(N, seed = 1)
#formula = random_formula()

test_atom : Formula = Atom(("p_0", 0))
test_atom_2 : Formula = Atom(("p_1",1))
test_formula : Formula = And(test_atom, test_atom_2)
test_trace : np.ndarray = sampled_tr[0]
p0_test_trace = test_trace[0,:]
p1_test_trace = test_trace[1, :]

test_atom_eval = test_atom.eval_trace(test_trace)
test_atom_2_eval = test_atom_2.eval_trace(test_trace)
test_formula_eval = test_formula.eval_trace(test_trace)

print(test_formula)
print(f"trace for p_0: {p0_test_trace}")
print(p0_test_trace.shape)
print(f"trace for p_1: {p1_test_trace}")
print(p1_test_trace.shape)
print(f"atom 1 valuation: {test_atom_eval}")
print(test_atom_eval.shape)
print(f"atom 2 valuation: {test_atom_2_eval}")
print(test_atom_2_eval.shape)
print(f"formula valuation: {test_formula_eval}")
print(test_formula_eval.shape)

#print(formula)
#print(formula.atoms())

end = time.time()
print(end - start)