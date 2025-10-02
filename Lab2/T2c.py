# Skriv en Python-funktion diskretisering_temperatur som returnerar systemmatrisen A (som en gles matris, dvs inte full) och högerledet HL (inklusive randvillkor)
# givet ett funktionshandtag för q(x) och värden på k, randvillkoren TL, TR och antal
# diskretiseringsintervall N. Ett anrop till funktionen kan se ut så här:
# A, HL = diskretisering_temperatur(N, q, k, Tl, Tr)
# Du kan testa din funktion genom att sätta N = 4 och verifiera att

#[[((b-a)/N)^2*q(x1))/k - T0], [(((b-a)/N)^2*q(x1))/k], [(((b-a)/N)^2*q(x1))/k -T4]]
import numpy as np

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
    print(A)
    A = k/h**2 * A  # skala med k/h^2
    #print("\n")
    #print("new matrix A")
    #print(A)
    # Högerled HL
    HL = np.array([q(xj) for xj in x[1:N]]) 
    HL[0] = HL[0] - (k / h**2) * TL
    HL[-1] = HL[-1] - (k / h**2) * TR

    
    return A, HL

# Exempel q(x)
def q(x):
    return 50 * x**3 * np.log(x + 1)

def main():
    k, TL, TR, N = 2, 2, 2, 4
    A, HL = diskretisering_temperatur(N, q, k, TL, TR)
    print("Systemmatris A:\n", A)
    print("Högerled HL:\n", HL)

main()
