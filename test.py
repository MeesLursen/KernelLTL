# from formula_class import Atom, Next, Formula

# AP = 3
# T = 10

# literal_cache = {
#     atom_idx: Atom(atom_idx)
#     for atom_idx in range(AP)
# }

# Chi: list[Formula] = []

# for t in range(T):
#     for i in range (AP):
#         formula = literal_cache[i]
#         for _ in range(t): 
#             formula = Next(formula)
#         Chi.append(formula)


# for phi in Chi:
#     print(phi)   

import math

eps     = 0.01
delta   = 1 - 0.99

N       = math.ceil((2 / eps**2) * math.log(2 * 1024 / delta))

print(N)