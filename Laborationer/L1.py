import numpy as np
import matplotlib.pyplot as plt
#a)
def U1a():
    def f(x, L):
        return (8/3) * (x / L) - 3 * (x / L)**2 + (1/3) * (x / L)**3 - (2/3) * np.sin(np.pi * x / L)

    L = 1

    x_vals = np.linspace(0, L, 100)
    y_vals = f(x_vals, L)

    plt.plot(x_vals, y_vals, label="f(x)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.show()

#b)svar: man kan hitta en. 
def U1b():

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

    x0 = 0.8
    tol = 1E-10
    root, iterations =fipforloop(1,x0,tol,1000)
    print('\n')
    print(f"Fixed point: {root:.8f}, after {iterations} iterations")

    def g_prim(x, L):
        return abs((((3*L)/8)* (6*(x/(L**2))) - (x**2)/(L**3) + ((2*np.pi)/(3*L)) * np.cos((np.pi*x)/L)))
    #print(g_prim(0.8,1))
    #print(g_prim(0.3,1))

#c)svar: använd U1b
def U1c():
    #samma som Ub()
    return 

#c)svar: hittar vid 0.3
def U1d():
    x0 = 0.2
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
    print(root)
    print(iterations)

#e)svar: båda konvergerar vid 0.8
def U1e():

    def newton_method(x0, tol, max_iter):
        f = lambda x:(8/3)*(x) - 3*(x)**2 + (1/3)*(x)**3 - (2/3)*np.sin(np.pi * x)
        f_prim = lambda x: (8/3) - 6*x + x**2 - (2*np.pi/3) * np.cos(np.pi*x)
        x = x0
        n_iter = 0
        Delta_x = tol + 1  
        Delta_x_n_list = []
        while Delta_x > tol:
            n_iter += 1
            xnew = x - (f(x)/(f_prim(x)))
            Delta_x = np.abs(xnew - x)
            #print(Delta_x)
            x = xnew
            #print(f"Iteration {n_iter}: x = {x}, x_new = {Delta_x}")
            Delta_x_n_list.append(Delta_x)
            if n_iter > max_iter:
                raise RuntimeError
        return Delta_x_n_list, n_iter
    
    def FPI_method(x0, tol, maxiter):
        g = lambda x: ((3/8)) * (3*(x)**2 - (1/3)*(x)**3 + (2/3)*np.sin(np.pi*x))
        Delta_x = tol + 1
        x = x0
        n_inter = 0
        Delta_x_FPI_list = []
        while Delta_x > tol:
            n_inter += 1
            xnew = g(x)
            Delta_x = np.abs(xnew - x)
            Delta_x_FPI_list.append(Delta_x)
            if n_inter > maxiter:
                raise RuntimeError
            else:
                x = xnew
        return Delta_x_FPI_list, n_inter

    x0 = 0.8
    tol = 1e-10

    Delta_x_newton,n_iter_n = newton_method(x0, tol, 1000)
    #print(Delta_x_newton)
    Delta_x_FPI, n_iter_FPI= FPI_method(x0, tol, 1000)

    #plt.plot(range(1, n_iter_n+1), Delta_x_newton,color = 'r')
    #plt.plot(range(1, n_iter_FPI+1),Delta_x_FPI,  color = 'b')
    plt.semilogy(range(1, n_iter_n+1), Delta_x_newton,color = 'r')
    #plt.semilogy(range(1, n_iter_FPI+1),Delta_x_FPI,  color = 'b')
    plt.xlabel('iterations')
    plt.ylabel('error')
    plt.grid(True)
    plt.show()

def U2a():
    return

def U2b():
    return

def U2c():
    return

def U2d():
    return

def U2e():
    return

def U2f():
    return

def U3a():
    return

def U3b():
    return

def U3c():
    return

def U3d():
    return

def U3e():
    return

def U3f():
    return

def U3g():
    return

#U1a()
#U1b()
#U1d()
#U1e()

#U2a()
#U2b()
#U2c()
#U2d()
#U2e()
#U2f()

#U3a()
#U3b()
#U3c()
#U3d()
#U3e()
#U3f()
#U3g()
