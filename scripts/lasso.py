from src.utils import *

def main(x, y):
    degree = 5
    grad_lasso = lambda th: gradient_lasso(th, x, y, lam=0.01, degree=degree)


    methods = ("plain", "momentum", "adagrad", "rmsprop", "adam")
    # methods = ("adam")
    max_iters = 100000 # må velge ganske lavt verdi her...? # divergerer for = 100

    iters = {}
    gam = np.logspace(-6,-4,10)
    for i in methods:
        iters[i] = []
        for j in gam:
            print(f"for gamma = {j}:")
            opt = optimise(grad_lasso, np.zeros(degree), i, j, num_iters = max_iters, tol=1e-6)
            if len(opt) >= max_iters:
                iters[i].append(np.nan)
            else:
                iters[i].append(len(opt))
    print(iters)

