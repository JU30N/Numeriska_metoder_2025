
import numpy as np
import numpy.polynomial.polynomial as poly
import os

def get_newton_interpolation_matrix(x):
    """
    Build the interpolation matrix for Newton interpolation.
    """
    n = len(x)
    matrix = np.zeros((n, n))
    matrix[:, 0] = 1  # First column is 1

    for j in range(1, n):
        for i in range(j, n):
            matrix[i, j] = matrix[i, j-1] * (x[i] - x[j-1])

    return matrix

def exempel_34_b(Tx=40):
    """
    Konstruera ett interpolationpolynom med Newtons ansats
    """
    # Mata in mätvärdena
    x = np.array([10, 30, 50, 70], dtype=float)
    y = np.array([1.308, 0.801, 0.549, 0.406], dtype=float)

    # coef = newton_divided_differences(x, y)
    # Beräkna Interpolationsmatrix
    A_newton = get_newton_interpolation_matrix(x)

    print('')
    b = y.reshape(len(y), 1)
    # Löser det linjära ekvationssystemet
    a = np.linalg.solve(A_newton, b)

    # print('\nKoefficienterna')
    # Koefficienterna skrivs ut
    # print(a)

    # print("\nInterpolation Matrix:")
    # print(A_newton)

    # To get interpolated values: y ≈ interpolation_matrix @ coefficients
    # Bilda det interpolerade polynomet
    def p(T): return a[0] + a[1]*(T-x[0]) + a[2]*(T-x[0]) * \
        (T-x[1]) + a[3]*(T-x[0])*(T-x[1])*(T-x[2])
    
    print(p)

    return p(Tx)[0]

mu_40_a = exempel_34_b(Tx=40)

def get_naiv_ansats():
    X_data = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
    Y_data = np.array([1.00, 0.7546, 0.5323, 0.3456, 0.1988])
    
    #Interpolering: Naiv ansats
    p = poly.polyfit(X_data, Y_data, 4)
    
    #Skriver ut interpolationspolynomet
    print('\nInterpoleringspolynomet är: ')
    print(p)

def fp(xi):
    f = lambda x: np.exp(-x)*np.cos(x)
    h = 1e-10
    return (f(xi+h)-f(xi-h))/2/h


def get_poly_centrerad(x,y):
    
    n = len(x)
    x_medel = np.mean(x)
    x_cent = x - x_medel
    #Interpolering: Naiv ansats
    p = poly.polyfit(x_cent, y, n-1)
    
    return p
def poly_centrerad():
    X_data = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
    Y_data = np.array([1.00, 0.7546, 0.5323, 0.3456, 0.1988])
    
    X_medel = np.mean(X_data)
    
    p = get_poly_centrerad(X_data, Y_data)
    #Skriver ut interpolationspolynomet
    print('\nInterpoleringspolynomet är: ')
    print(p)
        
    #Beräkna pprim(0.6)
    pprim = poly.polyder(p)
    dp_06 = poly.polyval(0.6-X_medel,pprim)
    print(f"\nb) dp/dx i 0.6 uppskattas = {dp_06}")
    #Calculate derivate of function: f(x) = exp(-x)*cos(x)
    #Läxa: Genomföra beräkningen manuellt.
    fp_exakt = fp(0.6)
    print(f"fprim(0.6) = {fp_exakt}")
        
    
def modell_funktion(x,a):
    f = a[0] + a[1]*x
    return f    
    

def minstakvadratmetoden(x,y):
        #Konstuera och lös normalekvationen
        
        n = len(x)
        x = np.array(x).reshape(n,1)
        y = np.array(y).reshape(n,1)
        
        A = np.hstack([x**0, x])
        AT = np.transpose(A)
        a = np.linalg.solve(AT@A,AT@y)
        
        #Beräkna 2-normen av residualvektorn
        r = A@a-y  #residualvektorn
        r_2norm = np.linalg.norm(r)
        SE = r_2norm**2
        RMSE = np.sqrt(SE/len(r))
        
        print(f"\n2-normen av r = {r_2norm}")
        print(f"\nSquare Error = {SE}")
        print(f"\nRoot Mean Square Error = = {RMSE}")
        
        return a
 
def get_mkm():
    xdata = [1,2,3,4]
    ydata = [2,3,2.5,5]   
    a = minstakvadratmetoden(xdata, ydata)
    print(a)


