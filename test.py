import numpy as np
from formula_class import Atom

# atom = Atom(("p_0", 0))

# print(atom.atoms())

rng = np.random.default_rng(seed = 1)

_UNARY_OPS = ['NOT', 'X', 'E', 'G']
_BINARY_OPS = ['AND', 'OR', 'IMPLIES', 'U']
_ALL_OPS = _UNARY_OPS + _BINARY_OPS
op = rng.choice(_ALL_OPS)
print(op)