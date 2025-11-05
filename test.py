import math
m       = 200

eps     = 0.01
delta   = 1 - 0.99
N       = math.ceil((2 / eps**2) * math.log(2 * m / delta))

print(N)