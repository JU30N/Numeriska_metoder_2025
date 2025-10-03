import numpy as np
import matplotlib.pyplot as plt


def diskretisering_temperatur(N, q, k, TL, TR, L=1.0):
    h = L / N
    x = np.linspace(0, L, N+1)  
    
    # Systemmatris A gles
    # A = np.zeros((N-1, N-1))#N-1 ggr N-1 matrix
    # for i in range(N-1):
    #     A[i, i] = -2
    #     if i > 0:
    #         A[i, i-1] = 1
    #     if i < N-2:
    #         A[i, i+1] = 1
    # A = k/h**2 * A  # skala med k/h^2
    #print("\n")
    #print("new matrix A")
    A = (k/h**2)*(-2*np.eye(N-1) + np.diag(np.ones(N-2),1) + np.diag(np.ones(N-2),-1))
    #print(A)
    # Högerled HL
    HL = np.array([q(xj) for xj in x[1:N]]) 
    #print(HL)
    HL[0] = HL[0] - (k / h**2) * TL
    HL[-1] = HL[-1] - (k / h**2) * TR
    #print(HL)
    # print(B)

    
    return x, A, HL

# Exempel q(x)
def q(x):
    return 50 * x**3 * np.log(x + 1)

def main():
    k, TL, TR, N = 2, 2, 2, 4
    x_want = 0.7
    N_values = [50, 100, 200, 400]
    T_values = []
    for N in N_values:
        x, A, HL = diskretisering_temperatur(N, q, k, TL, TR)
        T_solve = np.linalg.solve(A, HL)

        # Lägg till randvillkor
        wk = np.concatenate((TL,T_solve,TR), axis=None)

        
        T_at_x_want = np.interp(x_want, x, wk)
        T_values.append(T_at_x_want)
        print(f"N={N}, T({x_want}) : {T_at_x_want}")
    eh = T_values[0]
    eh_2 = T_values[1]
    eh_4 = T_values[2]
    eh_8 = T_values[3]
    steglangdshalvering_2 = np.abs(eh_2 - eh)/np.abs(eh_4 - eh_2)
    print(steglangdshalvering_2)
    steglangdshalvering_4 = np.abs(eh_4 - eh_2)/np.abs(eh_8 - eh_4)
    print(steglangdshalvering_4)
    #print(T_values)




main()