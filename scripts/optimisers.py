import numpy as np
from src.utils import *

def main(x, y):
    degree = 5
    X_norm = rescale_design_matrix(x, degree=degree)

    grads = {'ols': lambda th: gradient(th, x, y, lam=0.0, degree=degree), 'ridge': lambda th: gradient(th, x, y, lam=0.01, degree=degree)}


    eigenvalues_hessian_OLS = hessian_eigs(X_norm,lam = 0)
    lambda_max_OLS = np.max(eigenvalues_hessian_OLS)
    optimal_gamma = 0.9*(2/lambda_max_OLS)
    print(optimal_gamma)

    degree = 5

    methods = ("plain", "momentum", "adagrad", "rmsprop", "adam")

    max_iters = 10000

    print("For OLS:")
    iters = {}
    gam = np.logspace(-3,0,10)
    for i in methods:
        iters[i] = []
        for j in gam:
            opt = optimise(grads["ols"], np.zeros(degree), i, j, num_iters = max_iters)
            if len(opt) > max_iters:
                iters[i].append(np.nan)
            else:
                iters[i].append(len(opt))
    print(iters)


    print("For ridge:")
    iters = {}
    gam = np.logspace(-3,0,10)
    for i in methods:
        iters[i] = []
        for j in gam:
            opt = optimise(grads['ridge'], np.zeros(degree), i, j, num_iters = max_iters)
            if len(opt) > max_iters:
                iters[i].append(np.nan)
            else:
                iters[i].append(len(opt))
    print(iters)

    # vi har funnet ut: alle konvergerer for minst en gamma-verdi, utenom rmsprop. 
    # Vi burde se om optimal gamma analytisk stemmer overens med den beste gamma-verdien numerisk.