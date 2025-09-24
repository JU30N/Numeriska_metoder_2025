#Skriv ett Python-program som approximerar lösningen till ekvation (1) med Eulers
# metod-framåt och steglängden h = 0.1. Spara alla lösningsvärden (inklusive initialdata)
# i en vektor och plotta den numeriska lösningsvektorn som funktion av tiden. Skriv din
# kod så generellt som möjligt så att den går att återanvända för ett annat problem med
# annat högerled f(t, y) och initialdata.

#f(t,y) = 1 + t - y y(0) = 1. dy/dt = f(t,y) = 1 + t - y 
#t = T = 1.2

import numpy as np
import matplotlib.pyplot as plt
import os


def euler_system_forward_h(F, t0, tend, U0, h):
    
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
    y0 = np.array(U0, dtype=float)
    t_values = np.zeros(n_steps+1)
    y_values = np.zeros((n_steps+1, len(y0)))

    t_values[0] = t0
    y_values[0] = U0

    for i in range(n_steps):
        y_values[i+1] = y_values[i] + h * F(t_values[i], y_values[i])
        t_values[i+1] = t_values[i] + h

    return t_values, y_values

def plot_solutions(t_vals,y_vals):
    # Plot
    # plt.plot(t_vals, y_vals, label='y1(t) Euler')
    plt.plot(t_vals, y_vals, label='y2(t) Euler')
    
    plt.xlabel('t')
    plt.ylabel('U(t)')
    plt.legend()
    plt.title("Euler's Method for a System of ODEs")
    plt.grid(True)
    plt.show()  

def F(t, y):
    return np.array(1 + t - y)

def main():
    t0 = 0
    tend = 1.2
    U0 = [1]  #Initial values
    h = 0.1
    t_vals, y_vals = euler_system_forward_h(F, t0, tend, U0, h)
    plot_solutions(t_vals, y_vals)
main()
