from src.utils import *

def main(x, y):

    def scientific_label(value):
        coefficient, exponent = f"{value:.0e}".split("e")
        return rf"$\gamma={coefficient}\times 10^{{{int(exponent)}}}$"

    degree = 5
    lam = 0.01
    X = rescale_design_matrix(x, degree=degree)
    grad_lasso = lambda th: gradient_lasso(th, x, y, lam=lam, degree=degree)


    methods = ("momentum", "adagrad", "rmsprop", "adam", "plain")
    # methods = ("plain", "momentum")
    # methods = ("adam")
    max_iters = 100000
    iters = {}
    # g = np.logspace(-4,-1,10)
    g = np.logspace(-6,-4,10) # this is the original one
    for i in methods:
        plt.figure(figsize = (8,6))
        iters[i] = []
        for j in g:
            print(f"for gamma = {j}:")
            opt = optimise(grad_lasso, np.zeros(degree), i, j, num_iters = max_iters, tol=1e-8)
            if len(opt) >= max_iters:
                iters[i].append(np.nan)
            else:
                iters[i].append(len(opt))
            loss = np.mean((opt @ X.T - y) ** 2, axis=1) + lam * np.sum(np.abs(opt), axis=1)
            plt.plot(np.arange(len(opt)), loss, label=f"{scientific_label(j)}")
        # plt.title(f"Lasso convergence: {i}")
        # plt.tick_params(axis = "both", labelsize = 14)

        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # standard form
        # plt.tick_params(axis='both', labelsize=14)                      # tick numbers
        plt.rcParams['xtick.labelsize'] = 17
        plt.rcParams['ytick.labelsize'] = 16

        plt.xlabel("Iterations", fontsize = 18)
        plt.ylabel("Lasso objective", fontsize = 18)

        plt.yscale("log")
        plt.legend(fontsize=12)
        plt.grid(True, which="both", alpha=0.3)
        plt.savefig(f"figures/lasso/lasso_convergence_{i}.pdf", dpi=300, bbox_inches="tight")
    print(iters)






