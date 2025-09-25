#
#y' = [dq/dt, di/dt]
#
#F(t,Y) = [i, -(Ri/L) - (1/CL)q]
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def F(t, Y, R, L, C):
    #Y = [q, i]
    return [Y[1], -(R*Y[1]/L) - (1/(C*L))*Y[0]]




