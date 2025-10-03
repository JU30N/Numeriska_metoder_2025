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

    
    return A, HL

# Exempel q(x)
def q(x):
    return 50 * x**3 * np.log(x + 1)

def main():
    k, TL, TR, N = 2, 2, 2, 100
    A, HL = diskretisering_temperatur(N, q, k, TL, TR)
    print("Systemmatris A:\n", A)
    print("Högerled HL:\n", HL)

    T_solve = np.linalg.solve(A, HL)
    print("Lösning T:\n", T_solve)
    wk = np.concatenate((TL,T_solve,TR), axis=None)
    x = np.linspace(0, 1, N+1)
    plt.plot(x,wk)
    plt.xlabel("t")
    plt.grid(True)
    plt.show()

main()
