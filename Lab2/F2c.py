import numpy as np  
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
    
    
    n_steps = int(np.round(np.abs(tend-t0)/h))#använder round för att int avrundar neåt 
    t_values = np.zeros(n_steps+1)
    y_values = np.zeros(n_steps+1)

    t_values[0] = t0
    y_values[0] = U0

    for i in range(n_steps):
        y_values[i+1] = y_values[i] + h * F(t_values[i], y_values[i])
        t_values[i+1] = t_values[i] + h

    return t_values, y_values

def F(t, y):
    return 1 + t - y

def exakt_solution(t):
    return np.exp(-t) + t

def main():
    t0 = 0
    tend = 1.2
    U0 = 1 
    h = [0.2, 0.1, 0.05, 0.025, 0.0125]
    T = 1.2
    y_exakt = exakt_solution(T)
    errors = []
    p_values = []
    for step in h:
        t_vals, y_vals = euler_system_forward_h(F, t0, tend, U0, step)
        err = np.abs(y_vals[-1] - y_exakt)

        #print(f"Step: {step}, error value: {err}")
        errors.append(err)
    
    for i in range(1, len(errors)):
        p = np.log(errors[i-1]/errors[i]) / np.log(2)

        p_values.append(p)

    print(p_values)
        
    print(f"time: {tend - t0} s")
    
    
main()