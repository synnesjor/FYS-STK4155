import numpy as np
from src.utils import *
from matplotlib.ticker import LogFormatterSciNotation

def main(x, y):

    colours = ["#aa333c", "#99d2fb", "#9aab64", "#21534c", "#ef9d2c", "#B0a6df"]

    degree = 6
    X_norm = rescale_design_matrix(x, degree=degree)

    grads = {'ols': lambda th: gradient(th, x, y, lam=0.0, degree=degree), 'ridge': lambda th: gradient(th, x, y, lam = 0.0343, degree=degree)}


    eigenvalues_hessian_OLS = hessian_eigs(X_norm,lam = 0)
    eigenvalues_hessian_ridge = hessian_eigs(X_norm,lam = 0.0343)
    lambda_max_OLS = np.max(eigenvalues_hessian_OLS)
    lambda_max_ridge = np.max(eigenvalues_hessian_ridge)
    max_gamma_ols = 0.9*(2/lambda_max_OLS)
    max_gamma_ridge = 0.9*(2/lambda_max_ridge)
    print("The optimal gamma, found from the hessian matrix is", max_gamma_ols)
    print("The optimal gamma, found from the hessian matrix is", max_gamma_ridge)

    degree = 6

    methods = ("plain", "momentum", "adagrad", "rmsprop", "adam")

    max_iters = 100000

    # print("For OLS:")
    # iters = {}
    # gam = np.logspace(-4,1,10)
    # for i in methods:
    #     # print(f"For {i}")
    #     iters[i] = []
    #     for j in gam:
    #         opt = optimise(grads["ols"], np.zeros(degree), i, j, num_iters = max_iters, lasso_bool=False)
    #         if len(opt) > max_iters:
    #             iters[i].append(np.nan)
    #         else:
    #             iters[i].append(len(opt))
    # print(iters)


    print("For ridge:")
    iters = {}
    gam = np.logspace(-4,1,10)
    for i in methods:
        iters[i] = []
        for j in gam:
            opt = optimise(grads['ridge'], np.zeros(degree), i, j, num_iters = max_iters, lasso_bool=False)
            if len(opt) > max_iters:
                iters[i].append(np.nan)
            else:
                iters[i].append(len(opt))
    print(iters)

    # vi har funnet ut: alle konvergerer for minst en gamma-verdi, utenom rmsprop. 
    # Vi burde se om optimal gamma analytisk stemmer overens med den beste gamma-verdien numerisk.

    
    plt.figure(figsize=(8, 6))
    for idx, k in enumerate(methods):
        plt.loglog(gam,iters[k],label=k,marker="o",color=colours[idx])
    plt.gca().yaxis.set_major_formatter(LogFormatterSciNotation())
    plt.xlabel(r"Learning rate $\gamma$", fontsize=15)
    plt.ylabel("Number of iterations", fontsize=15)
    plt.grid(alpha=0.4)
    plt.axvline(max_gamma_ridge * (1 / 0.9), linestyle="dashed", color="#a9a9a9", label=r"2/$\lambda_{max}$")
    plt.legend(fontsize=13)
    plt.gcf().canvas.draw()
    plt.tick_params(axis='x', which='both', labelsize=15)
    plt.tick_params(axis='y', which='both', labelsize=15)
    plt.savefig("optimiser_algorithms_ridge_small_lambda.pdf", bbox_inches="tight")

    # plt.figure(figsize = (8,6))
    # for idx, k in enumerate(methods):
    #     plt.loglog(gam, iters[k], label = k, marker = "o", color = colours[idx])
    #     # plt.yscale("log")
    #     # plt.xscale("log")
    #     plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # standard form
    #             # plt.tick_params(axis='both', labelsize=14)                      # tick numbers
    #     plt.rcParams['xtick.labelsize'] = 17
    #     plt.rcParams['ytick.labelsize'] = 16
    #     plt.tick_params(axis='x', labelsize=12)
    #     plt.tick_params(axis='y', labelsize=12)
    #     plt.ylabel("number of iterations", fontsize = "14")
    #     plt.xlabel(rf"learning rate $\gamma$", fontsize = "14")
    #     plt.grid(alpha = 0.4)
    # plt.axvline(max_gamma_ridge*(1/0.9), linestyle = "dashed", color = "#a9a9a9", label = r"2/$\lambda_{max}$")
    # plt.legend(fontsize = "13")
    # plt.savefig("optimiser_algorithms_ridge_small_lambda.pdf")

