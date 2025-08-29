# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
#a)
def Ua():
    def f(x, L):
        return (8/3) * (x / L) - 3 * (x / L)**2 + (1/3) * (x / L)**3 - (2/3) * np.sin(np.pi * x / L)

    L = 1

    # Plotting the iterations
    x_vals = np.linspace(0, L, 100)
    y_vals = f(x_vals, L)

    plt.plot(x_vals, y_vals, label="f(x)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.show()

#b)svar: man kan hitta en. 
def Ub():

    def fipforloop(L, x0, tol, max_iter):
    # Initiera loopen
        g = lambda x, L: ((3/8) * L) * (3*(x/L)**2 - (1/3)*(x/L)**3 + (2/3)*np.sin(np.pi*x/L))
        x = x0
        for n in range(1, max_iter + 1):
                x_new = g(x, L)
                print(f"Iteration {n}: x = {x}, x_new = {x_new}")
                if np.abs(x_new - x) < tol:
                    return x_new, n
                x = x_new
        raise RuntimeError(
            "Fixed-point iteration did not converge within the maximum number of iterations.")

    x0 = 0.2
    tol = 1E-10
    root, iterations =fipforloop(1,x0,tol,1000)
    print('\n')
    print(f"Fixed point: {root:.8f}, after {iterations} iterations")

    def g_prim(x, L):
        return abs((((3*L)/8)* (6*(x/(L**2))) - (x**2)/(L**3) + ((2*np.pi)/(3*L)) * np.cos((np.pi*x)/L)))
    #print(g_prim(0.8,1))
    #print(g_prim(0.3,1))


def Uc():
    #samma som Ub()
    return 

def Ud():#svar den hittar båda då x0 är tillräckligt nära 
    x0 = 0.3
    tol = 1E-10
    max_iter = 1000

    def newton_method(x0, tol, max_iter):
        f = lambda x:(8/3)*(x) - 3*(x)**2 + (1/3)*(x)**3 - (2/3)*np.sin(np.pi * x)
        f_prim = lambda x: (8/3) - 6*x + x**2 - (2*np.pi/3) * np.cos(np.pi*x)
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
    
    root, iterations = newton_method(x0,tol,max_iter)
    print('\n')
    print(f"Fixed point: {root:.8f}, after {iterations} iterations")






#Ua()
#Ub()
Ud()