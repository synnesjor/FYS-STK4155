import numpy as np
import matplotlib.pyplot as plt
from src.utils import *

# Script for part 1b
def main(x, y):

    y = y - y.mean() # centre y

    # Choose degree
    degree = 5

    def ridge(x,y,lam,degree):
        X = rescale_design_matrix(x, degree)
        return ols(X,y) * (1/(1+lam))

    ridge_values = []
    lambda_values = np.logspace(-4, 2, 100)
    for each in lambda_values:
        ridge_values.append(ridge(x,y,each,degree))

    ridge_temp = np.array(ridge_values)

    ridge_values=ridge_temp.T

    plt.figure(figsize = (8,6))
    for i in np.arange(degree):
        plt.plot(lambda_values, ridge_values[i], label=f"$\\theta_{{{i+1}}}$")
    plt.xscale("log")
    plt.axhline(y=0, color = "gray", alpha = 0.2)
    plt.xlabel(f"$\\lambda$", fontsize = 18)
    plt.ylabel(f"$\\theta$", fontsize = 18)
    plt.xticks(size = 18)
    plt.yticks(size = 18)
    plt.grid(alpha = 0.3)
    plt.legend(fontsize = 16)
    plt.savefig('figures/lambda_values.pdf', dpi=300)

    print("Theta values for Ridge:")
    for each in ridge_values:
        print(each)

    # vi sammenlikner start theta-verdier for OLS og Ridge for å sjekke at det gir mening, vi burde starte på ca samme sted siden vi begynner med en liter verdi for lambda. 
    # deretter skal vi se hvordan theta verdiene for ridge utvikler seg, spesielt med tanke på hvordan de avhenger av lambda. 

    print(ridge_values)