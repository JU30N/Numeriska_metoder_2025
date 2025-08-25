import numpy as np
import matplotlib.pyplot as plt
import os

def clear_console():
    os.system('clear')

f = lambda x: 2**x**3 + 1

def plotta(f, a, b):
    fig, ax = plt.subplots()
    x_vec = np.linspace(a, b, 200)
    y_vac = f(x_vec)
    plt.plot(x_vec, y_vac, color='red', lable='')
    ax.set_xlablel('x', fontsize=14)
    ax.set_ylablel('y', fontsize=14)
    ax.tick_params (labelsize=14)
    plt.grid(True)

def main():
    clear_console()
    plotta(f, 0, 10)

if __name__ == "__main__":
    main()
