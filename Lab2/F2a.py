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
    
    
    n_steps = int(np.abs(tend-t0)/h)
    y0 = 1
    t_values = np.zeros(n_steps+1)
    y_values = np.zeros((n_steps+1, y0))

    t_values[0] = t0
    y_values[0] = 1

    for i in range(n_steps):
        y_values[i+1] = y_values[i] + h * F(t_values[i], y_values[i])
        t_values[i+1] = t_values[i] + h

    return t_values, y_values, tend

def F(t, y):
    return np.array(1 + t - y)

def main():
    t0 = 0
    tend = 1
    U0 = 1 # Initial values as a sequence
    h = [0.2, 0.1, 0.05, 0.025, 0.0125]
    for step in h:
        t_vals, y_vals, tend = euler_system_forward_h(F, t0, tend, U0, step)
        print(f"Step: {step}, value: {y_vals[-1]}")
        
    print(f"time: {tend - t0} s")
    
    
main()