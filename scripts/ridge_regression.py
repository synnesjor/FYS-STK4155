import numpy as np
import matplotlib.pyplot as plt
from src.utils import *

# Script for part 1b
def main(x, y):

    colours = ["#aa333c", "#9aab64", "#21534c", "#ef9d2c", "#B0a6df"]

    # Choose degree
    degree = 5

    ridge_values = []
    lambda_values = np.logspace(-4, 2, 100)
    for each in lambda_values:
        ridge_values.append(ridge(x,y,each,degree))

    ridge_temp = np.array(ridge_values)

    ridge_values=ridge_temp.T

    plt.figure(figsize = (8,6))
    for idx, i in enumerate(np.arange(degree)):
        plt.plot(lambda_values, ridge_values[i], label=f"$\\theta_{{{i+1}}}$", color = colours[idx])
    plt.xscale("log")
    plt.axhline(y=0, color = "gray", alpha = 0.2)
    plt.xlabel(rf" Penalty parameter $\lambda$", fontsize = 16)
    plt.ylabel(f"Polynomial coefficient $\\theta$", fontsize = 16)
    plt.xticks(size = 16)
    plt.yticks(size = 16)
    plt.grid(alpha = 0.4)
    plt.legend(fontsize = 16)
    plt.savefig('figures/lambda_values.pdf', dpi=300)

    print("Theta values for Ridge:")
    for each in ridge_values:
        print(each)

    # vi sammenlikner start theta-verdier for OLS og Ridge for å sjekke at det gir mening, vi burde starte på ca samme sted siden vi begynner med en liter verdi for lambda. 
    # deretter skal vi se hvordan theta verdiene for ridge utvikler seg, spesielt med tanke på hvordan de avhenger av lambda. 

    print(ridge_values)