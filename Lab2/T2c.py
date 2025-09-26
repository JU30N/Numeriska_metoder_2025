# Skriv en Python-funktion diskretisering_temperatur som returnerar systemmatrisen A (som en gles matris, dvs inte full) och högerledet HL (inklusive randvillkor)
# givet ett funktionshandtag för q(x) och värden på k, randvillkoren TL, TR och antal
# diskretiseringsintervall N. Ett anrop till funktionen kan se ut så här:
# A, HL = diskretisering_temperatur(N, q, k, Tl, Tr)
# Du kan testa din funktion genom att sätta N = 4 och verifiera att

#[[((b-a)/N)^2*q(x1))/k - T0], [(((b-a)/N)^2*q(x1))/k], [(((b-a)/N)^2*q(x1))/k -T4]]

import numpy as np  
import os

def diskretisering_temperatur(q, k, TL, TR, N):
    #A matrix 
    #HL med randvillkor
    

    h = 1/N
    
    HL = np.array([((h**2)*q(t))/k - TL], [((h**2)*q(t))/k], [((h**2)*q(t))/k - TR])

    A = np.array([-2,1,0 ], [1,-2,1 ], [0,1,-2 ])

    return A, HL 

def q(x):
    return 50 * (x**3) * np.log(x + 1)

def main():
    k, TL, TR, N = 2, 2, 2, 4
    diskretisering_temperatur(q, k, TL, TR, N)
    return

main()