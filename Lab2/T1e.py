
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
N = [20, 40, 80, 160]
U0 = [1,0]

for n in N:
    h = (b - a) / n
    t, Y = euler_system_forward_h(F, a, b, U0, h)
    print(Y)
    plt.plot(t, Y[:,0], label=f"N={n}")#vilken av q eller i som ska plottas?

sol = integrate.solve_ivp(F, [a,b], U0, method='RK45', t_eval=np.linspace(a,b,1000))
#print(sol.y)
plt.plot(sol.t, sol.y[0], color = "purple", label="sol")#sol.y[0] = q(t)

plt.xlabel("t")
plt.ylabel("q(t)")
plt.legend()
plt.grid(True)
plt.show()