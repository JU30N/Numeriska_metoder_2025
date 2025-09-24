import numpy as np
x0 = 2
tol = 1E-10
max_iter = 3

def newton_method(x0, tol, max_iter):
    f = lambda x:5*x + x**2 - np.cos(2*x)
    f_prim = lambda x: 5 + 2*x + 2*np.sin(2*x)
    x = x0
    n_iter = 0
    Delta_x = tol + 1  
    while Delta_x > tol:
        n_iter += 1
        xnew = x - (f(x)/(f_prim(x)))
        Delta_x = np.abs(xnew - x)
        print(Delta_x)
        x = xnew
        print(f"Iteration {n_iter}: x = {x}, x_new = {Delta_x}")


        if n_iter > max_iter:
            raise RuntimeError
    return x, n_iter

newton_method(x0, tol, max_iter)