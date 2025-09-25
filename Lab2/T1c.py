#
#y' = [dq/dt, di/dt]
#
#F(t,Y) = [i, -(Ri/L) - (1/CL)q]
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def F1(t, Y, R=1, L=2, C=0.5):
    #Y = [q, i]
    return [Y[1], -(R*Y[1]/L) - (1/(C*L))*Y[0]]
def F2(t, Y, R=0, L=2, C=0.5):
    #Y = [q, i]
    return [Y[1], -(R*Y[1]/L) - (1/(C*L))*Y[0]]


U0 = [1,0]
n_steps = 1000
a=0
b=20
#i)
#L = 2
#C = 0.5
#R = 1

sol1 = integrate.solve_ivp(F1, [a,b], U0, method='RK45', t_eval=np.linspace(a,b,n_steps))
sol2 = integrate.solve_ivp(F2, [a,b], U0, method='RK45', t_eval=np.linspace(a,b,n_steps))

plt.plot(sol1.t, sol1.y[0], color = "blue")
plt.plot(sol1.t, sol1.y[1], color = "red")
plt.plot(sol2.t, sol2.y[0], color = "green")
plt.plot(sol2.t, sol2.y[1], color = "orange")
plt.xlabel("t")
plt.grid(True)
plt.show()