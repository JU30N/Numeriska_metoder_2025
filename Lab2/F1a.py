#f(t,y) = 1 + t - y y(0) = 1. dy/dt = f(t,y) = 1 + t - y
#t = T = 1.2


#a
# Denna fil ritar ett riktningsfält för en differentialekvation på formen y' = f(x, y).

import numpy as np
import matplotlib.pyplot as plt



def plot_direction_field(f, y_range, t_range):
    y = np.linspace(y_range[0], y_range[1])
    t = np.linspace(t_range[0], t_range[1])
    Y, T = np.meshgrid(y, t)
    dt_dy = f(T, Y)
    U = np.ones(dt_dy.shape)
    V = dt_dy
    
    f_solve = lambda t: np.e**(-t) + t
    t_solution = np.linspace(t_range[0], t_range[1])
    y_solution = f_solve(t_solution)
    plt.plot(t_solution, y_solution, color = 'green')

    
    plt.quiver(T, Y, U, V, color='blue')

    plt.xlabel('t')
    plt.ylabel('y')
    
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.show()

f = lambda x,t: 1 + t - x


t_intervall = (0, 1.2)
y_intervall = (0, 5)


plot_direction_field(f, y_intervall, t_intervall)