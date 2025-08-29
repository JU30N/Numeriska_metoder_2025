import numpy as np






# funktion for g(x)
def gfun(x):    
    g = 1/3*(x**2 + 1)
    return g
x0 = 0.5 # Startgissning
it = 0 # Raknare for antal iterationer
maxiter = 100 # Max antal iterationer
diffv = 1 # For att initiera while-slingan
tol = 1E-8 # Tolerans
x = x0 # Initiera x
while diffv > tol and it < maxiter:#differens mellan nya - gamla värdet > tol => Stop, max antal iterationer
# Uppdatera vardet med fixpunkt
    xnew = gfun(x)
    # For avbrottsvillkoret
    diffv = np.abs(xnew - x)
    # Uppdatera gamla vardet pa x
    x = xnew
    # Uppdatera raknare
    it =+ 1
    # Skriver ut resultatet
    print([it,x,diffv])


# x = np.linspace(0,7,100)      #vektor med x-värden   #f(x)
# y = gfun(x)                     #vektor med y-värden
# fig, ax = plt.subplots()      #skapa instanserna fig och ax
# ax.plot(x,y)
# xsol = 0.7390851332
# ax.scatter(xsol,f(xsol), color="red")
# #plot egenskaper
# ax.tick_params(labelsize=14)
# plt.grid(True,linestyle='-.')
# #plt.ylim([0, 60])
# plt.xlabel('x')
# plt.ylabel('f(t)=cos(x)-x')
# plt.show()