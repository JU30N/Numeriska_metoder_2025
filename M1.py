
import numpy as np
import os

def clear_console():
    os.system('clear')
    
def func(x):
    f = x**3 +4*x + 3
    fprim = 3*x**2 + 4
    return f, fprim

def f(x):
    f= x
    return f
    
    
def NewtonWhileLoop(func, x0, tol, max_iter):

    """
    Parameter:
        func        : tuple with f(x) and fprim(x) 
        xo          : initial guess
        tol         : Tolerans for convergence
        max_iter    : the max numbers of iterations

    Output:
        x           : root
        n           : number of iterations used             
    """

    # Initiera loopen
    x = x0
    DeltaX = tol + 1.0
    n = 0
    while DeltaX > tol:
        n += 1
        f, fp = func(x)
        xnew = x - f/fp
        DeltaX = np.abs(xnew-x)
        x = xnew
        if n > max_iter:
          raise RuntimeError(
              "Fixed-point iteration did not converge within the maximum number of iterations.")
    return x, n 


def secantWithForLoop(f, x0, x1, tol, max_iter):
    """
    Finds a root of the function f using the Secant Method.

    Parameters:
        f         : function for which we seek f(x) = 0
        x0, x1    : initial guesses
        tol       : tolerance for convergence
        max_iter  : maximum number of iterations

    Returns:
        root approximation and list of iterates
    """
    iterates = [x0, x1]

    for i in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)
        
        #Granska villkoret f(xi+1) != f(xi)
        if fx1 - fx0 == 0:
            print("Zero division in denominator — stopping iteration.")
            return None, iterates

        # Sekantformeln
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        iterates.append(x2)

        # Trace output
        print(f"Iteration {i+1}: x = {x2:.8f}, f(x) = {f(x2):.2e}")

        #Stoppvillkoret
        if abs(x2 - x1) < tol:
            break

        #Uppdatera iterationen
        x0 = x1
        x1 = x2
        #x0, x1 = x1, x2

    return x2, iterates


def fip(g, x0, tol, max_iter):
    """
    Parameter:
        g(x)        : The function handle 
        xo          : initial guess
        tol         : Tolerans for convergence
        max_iter     : the max numbers of iterations

    Output:
        x           :fixed-point
        n           : number of iterations used             
    """

    # Initiera loopen
    x = x0
    DeltaX = tol + 1.0
    n = 0
    while DeltaX > tol:
        n += 1
        xold = x
        x = g(xold)
        DeltaX = np.abs(x-xold)
        if n > max_iter:
          raise RuntimeError(
              "Fixed-point iteration did not converge within the maximum number of iterations.")
    return x, n

def fip_analys(g, x0, tol, max_iter):
    #Initiera flera tomma listor for konvergence analyser
    x_i = []
    diffvec_fip = []
    x = x0
    DeltaX = tol + 1.0
    n = 0
    np.append(diffvec_fip,DeltaX)
    x_i = np.append(x_i,x0)
    while DeltaX > tol:
        n += 1
        xold = x
        x = g(xold)
        DeltaX = np.abs(x-xold)
        #diffvec_fip = np.append(diffvec_fip,DeltaX)
        np.append(diffvec_fip,DeltaX)
        x_i = np.append(x_i,x)
        #print('')
        print(n, x , g(x), DeltaX)
        if n > max_iter:
          raise RuntimeError(
                "Fixed-point iteration did not converge within the maximum number of iterations.")
        
          
    return x, n, x_i

def kanalys_fip(g,xi,root,n):  
    #Konvergens i fixpunktsmetoden
    xi = np.array(xi)
    gi = np.array(g(xi))
    ei = np.array(np.abs(xi-root))
    eikvot = []
    eikvot = np.append(eikvot,1.0)
    for i in range(1,n+1):
        kvot = ei[i]/ei[i-1]
        eikvot = np.append(eikvot, kvot)   
    table = np.transpose(np.vstack([xi,gi,ei,eikvot]))
    dpTable = pd.DataFrame(table)
    print('')
    print(dpTable)

def NewtonAnalys(func, x0, tol, max_iter):
    """
    Parameter:
        func        : tuple with f(x) and fprim(x) 
        xo          : initial guess
        tol         : Tolerans for convergence
        max_iter    : the max numbers of iterations

    Output:
        x           : root
        n           : number of iterations used 
        xi            
    """

    # Initiera loopen
    x_i = []
    x = x0
    DeltaX = tol + 1.0
    n = 0
    print(f"{n},{x:.15f},{DeltaX:.8e}")
    x_i = np.append(x_i,x0)
    while DeltaX > tol:
        n += 1
        f, fp = func(x)
        xnew = x - f/fp
        DeltaX = np.abs(xnew-x)
        x = xnew
        x_i = np.append(x_i,x)
        print('')
        print(f"{n},{x:.15f},{DeltaX:.8e}")
        if n > max_iter:
          raise RuntimeError(
              "Fixed-point iteration did not converge within the maximum number of iterations.")
    return x, n, x_i    

def kanalys(func,xi,root,n):  
    #Konvergens i fixpunktsmetoden
    f, fp = func(xi)
    xi = np.array(xi)
    fi = np.array(f)
    ei = np.array(np.abs(xi-root))
    eikvot = []
    eikvot = np.append(eikvot,1.0)
    for i in range(1,n+1):
        kvot = ei[i]/ei[i-1]**2
        eikvot = np.append(eikvot, kvot)   
    table = np.transpose(np.vstack([xi,fi,ei,eikvot]))
    print('')

root, iterations, xi = fip_analys(g,x0,tol,100)
kanalys_fip(g,xi,root,iterations)


root, iterations, xi = NewtonAnalys(func,x0,tol,max_iter)
kanalys(g,xi,root,iterations)