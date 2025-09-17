import numpy as np
import matplotlib.pyplot as plt
# Givna data
t = np.array([0,1/4,1/2,3/4])
y = np.array([1,3,2,1/2])

def minstakvadratmetoden(t, y):
    n = len(t)
    y = y.reshape(n,1)

    # Bygg designmatrisen
    A = np.column_stack([np.ones_like(t), np.cos(2*np.pi*t), np.sin(2*np.pi*t)])

    # Lös normalekvationerna
    AT = A.T
    a = np.linalg.solve(AT @ A, AT @ y)

    return a

B = np.matmul(A.transpose(), A)#multiplicera matrisen AT med A 
b = np.matmul(A.transpose(), y)

C = np.linalg.solve(B, b)
