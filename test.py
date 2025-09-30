import numpy as np
from formula_class import Atom

# atom = Atom(("p_0", 0))

# print(atom.atoms())

trace = np.asarray([0,0,0,0,1,0,1,1,1,1],dtype=bool)
T = trace.shape[0]

trace[-1] = False
print(trace)

