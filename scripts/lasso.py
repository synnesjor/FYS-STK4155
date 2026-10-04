from src.utils import *

def main(x, y):
    degree = 5
    lam = 0.01
    X = rescale_design_matrix(x, degree=degree)
    grad_lasso = lambda th: gradient_lasso(th, x, y, lam=lam, degree=degree)


    methods = ("plain", "momentum", "adagrad", "rmsprop", "adam")
    # methods = ("adam")
    max_iters = 100000
    iters = {}
    g = np.logspace(-6,-4,10)
    for i in methods:
        plt.figure()
        iters[i] = []
        for j in g:
            print(f"for gamma = {j}:")
            opt = optimise(grad_lasso, np.zeros(degree), i, j, num_iters = max_iters, tol=1e-8)
            if len(opt) >= max_iters:
                iters[i].append(np.nan)
            else:
                iters[i].append(len(opt))
            loss = np.mean((opt @ X.T - y) ** 2, axis=1) + lam * np.sum(np.abs(opt), axis=1)
            plt.plot(np.arange(len(opt)), loss, label=f"gamma={j:.1e}")
        plt.title(f"Lasso convergence: {i}")
        plt.xlabel("Iterations")
        plt.ylabel("Lasso objective")
        plt.yscale("log")
        plt.legend(title="Learning rate", fontsize=8)
        plt.grid(True, which="both", alpha=0.3)
        plt.savefig(f"figures/lasso/lasso_convergence_{i}.pdf", dpi=300, bbox_inches="tight")
    print(iters)

