# Verifiera att din lösning med Eulers metod-framåt vid tid t = T, ger felet ek = |yk(T) − yexakt(T)| ≈ 0.0188.
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import integrate

# För fallet med dämpad svängning, 
# utför en konvergensstudie för Euler framåt och bestäm
# noggrannhetsordningen för metoden empiriskt. 
# Beräkna felet komponentvis, se förklaring
# nedan, vid sluttiden T = 20 och använd lösningen från solve_ivp som referenslösning. Gör
# så här: Börja med ett värde på N som leder till en stabil numerisk lösning. Dubblera sedan
# N (halvera tidssteget h) successivt och beräkna felen (ett fel per komponent i lösningen) för
# varje värde på N. Följ stegen i F2 c) för att beräkna noggrannhetsordningen komponentvis

def euler_system_forward(F, t0, tend, U0, h):
    
    """ 
    Implements Euler's method forward for a system of ODEs.
    
    Parameters:
        F       : function(t, U) → dU/dt (returns numpy array)
        t0      : initial time
        tend    : Final time
        U0      : initial state (numpy array)
        h       : step size
    
    Returns:
        t_values : numpy array of time points
        y_values : numpy array of state values (n_steps+1 x len(y0))
    """
    
    
    n_steps = int(np.abs(tend-t0)/h)
    t_values = np.zeros(n_steps+1)
    y_values = np.zeros((n_steps+1, len(U0)))
    #print(y_values)
    t_values[0] = t0
    y_values[0] = U0

    for i in range(n_steps):
        y_values[i+1] = y_values[i] + h * F(t_values[i], y_values[i])
        t_values[i+1] = t_values[i] + h

    return t_values, y_values



def F(t, Y, L = 2, C = 0.5, R = 1):
    return np.array([Y[1], -(R*Y[1]/L) - (1/(C*L))*Y[0]])

a, b= 0, 20
N = [40, 80, 160]
U0 = [1,0]
errors = []
hs = []

for n in N:
    h = (b - a) / n
    t, Y = euler_system_forward(F, a, b, U0, h)
    y_T = Y[-1]  # sol @ T
    sol_ref = integrate.solve_ivp(F, [a,b], U0, t_eval=[b])
    y_ref_T = sol_ref.y[:, -1]#sista vid T
    err = np.linalg.norm(y_T - y_ref_T) 
    errors.append(err)
    hs.append(h)

for k in range(len(errors)-1):
    p = np.log(errors[k]/errors[k+1]) / np.log(2)
    print(f"Between N={N[k]} and N={N[k+1]}: p ≈ {p:.2f}")
