# Verifiera att din lösning med Eulers metod-framåt vid tid t = T, ger felet ek = |yk(T) − yexakt(T)| ≈ 0.0188.
import numpy as np
import matplotlib.pyplot as plt
import os
# Detta program löser en differentialekvation med Eulers metod framåt,
# och verifierar felet vid en specifik tidpunkt T.

import numpy as np
import matplotlib.pyplot as plt

def euler_forward(t0, y0, tend, h):
    f = lambda t,y: 1 + t - y
    num_steps = int(round((tend - t0) / h))
    t_vals = np.zeros(num_steps + 1)
    y_vals = np.zeros(num_steps + 1)
    
    t_vals[0] = t0
    y_vals[0] = y0

    for i in range(num_steps):
        t_vals[i+1] = t_vals[i] + h
        y_vals[i+1] = y_vals[i] + h * f(t_vals[i], y_vals[i])

    return t_vals, y_vals

def exact_solution(t):
    return np.e**(-t) + t
T = 1.2
t0 = 0.0
y0 = 1.0
tend = T
h = 0.1

t_vals, y_vals = euler_forward(t0, y0, tend, h)

t_at_T = t_vals[-1]
y_at_T = y_vals[-1]

y_at_T_exact_sol= exact_solution(t_at_T)

error = np.abs(y_at_T - y_at_T_exact_sol)

print(f"Den numeriska lösningen vid t = {t_at_T:.1f} är: {y_at_T:.8f}")
print(f"Den exakta lösningen vid t = {t_at_T:.1f} är: {y_at_T_exact_sol:.8f}")
print(f"Felet ek vid t = {t_at_T:.1f} är: {error:.4f}")