import numpy as np
from fractions import Fraction

A = np.array([[1, 0.5, 1/3, 1/4],
              [1, 0.25, 0.25**2, 0.25**3],
              [1, 0.75, 0.75**2, 0.75**3],
              [1, 1, 1, 1]])

b = np.array([0.5, -1, -1.5, 2])

x = np.linalg.solve(A, b)
print(x)
