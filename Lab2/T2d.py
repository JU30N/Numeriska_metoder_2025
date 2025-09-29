import numpy as np

def diskretisering_temperatur(q, k, TL, TR, N, L=1.0):
    h = L / N
    x = np.linspace(0, L, N+1)
    
    # Systemmatris A
    A = np.zeros((N-1, N-1))
    for i in range(N-1):
        A[i, i] = -2 * k / h**2
        if i > 0:
            A[i, i-1] = k / h**2
        if i < N-2:
            A[i, i+1] = k / h**2
    
    # Högerled HL
    HL = np.array([q(xj) for xj in x[1:N]], dtype=float)
    HL = (h**2 / k) * HL
    
    # Justera för randvillkor
    HL[0] -= TL
    HL[-1] -= TR
    
    return A, HL

# Exempel q(x)
def q(x):
    return 50 * x**3 * np.log(x + 1)

# Test
def main():
    k, TL, TR, N = 2, 2, 2, 4
    A, HL = diskretisering_temperatur(q, k, TL, TR, N)
    print("Systemmatris A:\n", A)
    print("Högerled HL:\n", HL)

main()
