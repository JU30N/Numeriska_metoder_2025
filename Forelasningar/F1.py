import numpy as np
import matplotlib.pyplot as plt
import os

def clear_console():
    os.system('clear')

f = lambda x: 2*x**3 + 1

def plotta(f, a, b):
    fig, ax = plt.subplots()
    x_vec = np.linspace(a, b, 200)
    y_vac = f(x_vec)
    plt.plot(x_vec, y_vac, color='red', label='')
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.tick_params (labelsize=14)
    plt.show()

def main():
    clear_console()
    plotta(f, 0, 10)

if __name__ == "__main__":
    main()
