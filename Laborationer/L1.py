import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import os

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
    #labb fråga behöver man ta fram varenda polynom för att det är ju ändå 
    # samma bara fel om datorn inte kan räkna ut 

    #Naiv: 11
    #centrerad: 11
    #Newton: 11

    X_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    Y_data = np.array([421, 553, 709, 871, 1021, 1109, 1066, 929, 771, 612, 463, 374])
    p = poly.polyfit(X_data, Y_data, 11)#makes a polynom using Naiv
    x = np.linspace(0, 12, 1000)
    y = poly.polyval(x, p)
    fig, ax = plt.subplots()
    ax.plot(x, y, 'b')
    plt.grid(True)
    plt.show()

    def newton_interpolation():

        def newton_interpolation_matrix(x):    
            n = len(x)
            matrix = np.zeros((n, n))
            matrix[:, 0] = 1  # First column is 1s
            for j in range(1, n):
                for i in range(j, n):
                    matrix[i, j] = matrix[i, j-1] * (x[i] - x[j-1])
            return matrix
        print('')
        b = Y_data.reshape(len(Y_data),1)#makes it to a column vector
        print(b)
        a = np.linalg.solve(newton_interpolation_matrix(X_data), b)
        print(a)
        print('')


        print("\nInterpolation Matrix:")
        print(newton_interpolation_matrix(X_data))

        p = lambda T: a[0] + a[1]*(T-X_data[0]) + a[2]*(T-X_data[0])*(T-X_data[1])


    newton_interpolation()
    #alla olika kommer visa samma polynom bara på olika sätt
    
def U2b():
    #förstår inte andra delen av uppgiften
    X_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    Y_data = np.array([421, 553, 709, 871, 1021, 1109, 1066, 929, 771, 612, 463, 374])

    vander_matrix = np.vander(Y_data)#vander matrix from allt the 

    def get_newton_interpolation_matrix(x):
        n = len(x)
        matrix = np.zeros((n, n))
        matrix[:, 0] = 1  # First column is 1

        print(matrix)

        for j in range(1, n): #for j in range(1, 12)
            for i in range(j, n):#for i in range(j, 12)
                matrix[i, j] = matrix[i, j-1] * (x[i] - x[j-1])# matrix[2, 1] * (x[2] - x[2-1]) => 1 * (709 - 553)
        print(matrix)
        return matrix

    def get_poly_centrerad_matrix(x):
        #print(x)
        n = len(x)
        x_medel = np.mean(x)#medelvärdet
        x_cent = x - x_medel#central alltså tm
        #print(x_cent)
        matrix = np.vander(x_cent, n)
        #print(matrix)
        return matrix

    print(f"konditionstal Naiv: {np.linalg.cond(np.vander(X_data), p = np.inf)}")
    print("\n")
    print(f"konditionsral Newton: {np.linalg.cond(get_newton_interpolation_matrix(X_data), p = np.inf)}")
    print("\n")
    print(f"konditionstal Center: {np.linalg.cond(get_poly_centrerad_matrix(X_data), p = np.inf)} ")

    avrundningsfel = .0

def U2c():
    X_data = np.array([4, 5, 6, 7, 8])
    Y_data = np.array([871, 1021, 1109, 1066, 929])  

    def minstakvadratmetoden(x, y):
        n = len(x)
        x = np.array(x).reshape(n,1)
        y = np.array(y).reshape(n,1)
        
        # matris
        A = np.hstack([x**0, x, x**2])
        AT = A.T
        
        # normal ekvation lösning
        a = np.linalg.solve(AT@A, AT@y)
        
        # resudial vektorn
        r = A@a - y  
        r_2norm = np.linalg.norm(r)
        SE = r_2norm**2
        RMSE = np.sqrt(SE/len(r))
        
        print(f"\n2-normen av r = {r_2norm}")
        print(f"\nSquare Error = {SE}")
        print(f"\nRoot Mean Square Error = {RMSE}")
        
        return a
    minstakvadratmetoden(X_data, Y_data)

def U2d():
    X_data = np.array([4, 5, 6, 7, 8])
    Y_data = np.array([871, 1021, 1109, 1066, 929])

    def minstakvadratmetoden(x, y):
        n = len(x)
        x = np.array(x).reshape(n,1)
        y = np.array(y).reshape(n,1)
        
        # matris
        A = np.hstack([x**0, x, x**2, x**3])
        AT = A.T
        
        # normal ekvation lösning
        a = np.linalg.solve(AT@A, AT@y)
        
        # resudial vektorn
        r = A@a - y  
        r_2norm = np.linalg.norm(r)
        SE = r_2norm**2
        RMSE = np.sqrt(SE/len(r))
        
        print(f"\n2-normen av r = {r_2norm}")
        print(f"\nSquare Error = {SE}")
        print(f"\nRoot Mean Square Error = {RMSE}")
        
        return a
    minstakvadratmetoden(X_data, Y_data)

def U2e():
    X_data = np.array([4, 5, 6, 7, 8])
    Y_data = np.array([871, 1021, 1109, 1066, 929])
    def minstakvadratmetoden(x,y):
        n = len(x)
        x = np.array(x).reshape(n,1)
        y = np.array(y).reshape(n,1)

        omega = 2*np.pi/12
        A = np.hstack([x**0, np.cos(omega*x), np.sin(omega*x)])
        AT = np.transpose(A)
        a = np.linalg.solve(AT@A, AT@y)

        # residualer
        r = A@a - y
        r_2norm = np.linalg.norm(r)
        SE = r_2norm**2
        RMSE = np.sqrt(SE/len(r))

        print(f"2-normen av r = {r_2norm}")
        print(f"Square Error = {SE}")
        print(f"Root Mean Square Error = {RMSE}")

        return a
    minstakvadratmetoden(X_data, Y_data)
    

def U2f():
    return

def U3a():
    def trapetskvadratur(f,n, v_indata):

        h = (v_indata[1] - v_indata[0])/n                 # n trapets
        x = np.linspace(v_indata[0],v_indata[1],n+1)    # a och b ingår i intervallet
        y = f(x)
        I_T = h*(y[0] + 2*np.sum(y[1:-1]) + y[-1])/2
        return I_T

    n = 1000#antal trapets
    indata = np.array([0, 2])
    f = lambda x:(x**3)*(np.e**x)
    print(trapetskvadratur(f,n,indata))


def U3b():

    def trapetskvadratur(f,n, v_indata):
        h = (v_indata[1] - v_indata[0])/n                 # n trapets
        x = np.linspace(v_indata[0],v_indata[1],n+1)    # a och b ingår i intervallet
        y = f(x)
        I_T = (h*(y[0] + 2*np.sum(y[1:-1]) + y[-1]))/2
        e_h = np.abs(20.7781-I_T)
        # print(I_T)
        # print(e_h)
        print(n)
        #steglängds halvering är att halvera h = (a-b)/n genom att ändra på n så att h/2 = (a-b)/2n
        return I_T

    n = 1000#antal trapets
    indata = np.array([0, 2])
    f = lambda x:(x**3)*(np.e**x)
    I_t = trapetskvadratur(f,n,indata)
    I_t_half = trapetskvadratur(f,(2*n),indata)
    I_t_fouth = trapetskvadratur(f,(4*n),indata)

    print(I_t)
    print(I_t_half)
    print(I_t_fouth)

    steglangdshalvering = (np.abs(I_t_half - I_t)) / (np.abs(I_t_fouth - I_t_half))
    print("steglängshalvering : " + str(steglangdshalvering))

def U3c():
    def trapetskvadratur(n, v_indata):

        h = (v_indata[1] - v_indata[0])/n                 # n trapets h = 1
        y = [12.00, 15.10, 19.01, 23.92, 30.11, 37.90, 47.70, 60.03, 75.56]
        I_T = h*(y[0] + 2*np.sum(y[1:-1]) + y[-1])/2
        return I_T

    n = 8#antal trapets
    indata = np.array([2014, 2022])
    
    print("skattade värde på kW " + str(trapetskvadratur(n,indata)))

def U3d():
    def trapetskvadratur(n, v_indata):

        h = (v_indata[1] - v_indata[0])/n                 # n trapets
        y = [12.00, 15.10, 19.01, 23.92, 30.11, 37.90, 47.70, 60.03, 75.56]
        I_T = h*(y[0] + 2*np.sum(y[1:-1]) + y[-1])/2
        return I_T
    def trapetskvadratur_2(n, v_indata):

        h = (v_indata[1] - v_indata[0])/(2*n)                 # n trapets
        y = [12.00, 19.01, 30.11, 47.70, 75.56]
        I_T = h*(y[0] + 2*np.sum(y[1:-1]) + y[-1])/2
        return I_T
    def trapetskvadratur_4(n, v_indata):

        h = (v_indata[1] - v_indata[0])/(4*n)                 # n trapets
        y = [12.00, 30.11, 75.56]
        I_T = h*(y[0] + 2*np.sum(y[1:-1]) + y[-1])/2
        return I_T
    def trapetskvadratur_8(n, v_indata):

        h = (v_indata[1] - v_indata[0])/(8*n)                 # n trapets
        y = [12.00, 75.56]
        I_T = h*(y[0] + 2*np.sum(y[1:-1]) + y[-1])/2
        return I_T
    n = 8#antal trapets
    indata = np.array([2014, 2022])
    error_1 = np.abs(trapetskvadratur(n, indata) - trapetskvadratur_2(n, indata))
    error_2 = np.abs(trapetskvadratur_2(n, indata) - trapetskvadratur_4(n, indata))
    error_3 = np.abs(trapetskvadratur_4(n, indata) - trapetskvadratur_8(n, indata))
    print("eh1 : " + str(error_1))
    print("eh2 : " + str(error_2))
    print("eh3 : " + str(error_3))
    print(error_1 / error_2)

def U3e():

    def trapets(y_data, h):
        return h * (y_data[0]/2 + np.sum(y_data[1:-1]) + y_data[-1]/2)
    
    def simpsons(y_data, h):
        if (len(y_data) - 1) % 2 != 0:
            raise ValueError("")
        return h/3 * (y_data[0] + 4*np.sum(y_data[1:-1:2]) + 2*np.sum(y_data[2:-1:2]) + y_data[-1])
    
    y = np.array([12.00, 15.10, 19.01, 23.92, 30.11, 37.90, 47.70, 60.03, 75.56])
    h = 1.0  
    T_h = trapets(y, h)
    y_2h = y[::2]  # varannan data punkt
    T_2h = trapets(y_2h, 2 * h)
    r_e = (4*T_h - T_2h) / 3
    print(r_e)
    I_s= simpsons(y, h)
    print(I_s)


def U3f():
    x_data = np.array([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])

    y_data = np.array([12.00, 15.10, 19.01, 23.92, 30.11, 37.90, 47.70, 60.03, 75.56])

    x_new = x_data - 2014
    Y_new = np.log(y_data)

    A = np.vstack([np.ones(len(x_new)), x_new]).T
    print(A)
    koefficienter, residualer, rang, singulärvärden = np.linalg.lstsq(A, Y_new, rcond=None)

    ln_a = koefficienter[0]
    b = koefficienter[1]
    a = np.exp(ln_a)


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
U3f()
#U3g()
