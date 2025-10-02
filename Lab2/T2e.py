import numpy as np
import matplotlib.pyplot as plt


def diskretisering_temperatur(N, q, k, TL, TR, L=1.0):
    h = L / N
    x = np.linspace(0, L, N+1)  
    
    # Systemmatris A gles
    A = np.zeros((N-1, N-1))#N-1 ggr N-1 matrix
    #print(A)
    for i in range(N-1):
        A[i, i] = -2
        if i > 0:
            A[i, i-1] = 1
        if i < N-2:
            A[i, i+1] = 1
    #print("\n")
    #print("matrix A")
    #print(A)
    A = k/h**2 * A  # skala med k/h^2
    #print("\n")
    #print("new matrix A")
    #print(A)
    # Högerled HL
    HL = np.array([q(xj) for xj in x[1:N]]) 
    HL[0] = HL[0] - (k / h**2) * TL
    HL[-1] = HL[-1] - (k / h**2) * TR


    return x, A, HL

# Exempel q(x)
def q(x):
    return 50 * x**3 * np.log(x + 1)

def konvergensstudie(q, k, TL, TR, x_want=0.7):
    N_values = [50, 100, 200, 400]
    T_values = []

    for N in N_values:
        x, A, HL = diskretisering_temperatur(N, q, k, TL, TR)
        T_solve = np.linalg.solve(A, HL)

        # Lägg till randvillkor
        T_full = np.zeros(N+1)
        T_full[0] = TL
        T_full[-1] = TR
        T_full[1:-1] = T_solve

        # Interpolera värdet vid x_eval
        T_at_x = np.interp(x_want, x, T_full)
        T_values.append(T_at_x)
        print(f"N={N}, T({x_want}) ≈ {T_at_x}")



konvergensstudie(q,2, 2,2)
