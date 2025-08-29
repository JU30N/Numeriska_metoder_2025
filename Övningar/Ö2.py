import numpy as np
from scipy.linalg import lu

# Example matrix
A = np.array([
    [2, 3, 1, 5],
    [6, 13, 5, 19],
    [2, 19, 10, 23],
    [4, 10, 11, 31]
], dtype=float)

# LU factorization
P, L, U = lu(A)
print("Check:", np.allclose(P @ L @ U, A))
