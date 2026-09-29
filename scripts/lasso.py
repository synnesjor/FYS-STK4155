from src.utils import *

def main(x, y):
    degree = 5
    grad_lasso = lambda th: gradient_lasso(th, x, y, lam=0.1, degree=degree)


    methods = ("plain", "momentum", "adagrad", "rmsprop", "adam")
    max_iters = 1000 # må velge ganske lavt verdi her...? # divergerer for = 100

    print("For lasso:")
    iters = {}
    gam = np.logspace(-3,0,5)
    for i in methods:
        iters[i] = []
        for j in gam:
            opt = optimise(grad_lasso, np.zeros(degree), i, j, num_iters = max_iters)
            if len(opt) >= max_iters:
                iters[i].append(np.nan)
            else:
                iters[i].append(len(opt))
    print(iters)