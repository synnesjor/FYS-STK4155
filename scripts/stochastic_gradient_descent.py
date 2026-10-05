""" 
    LLM ASSISTED: 
    
    Tools: Chat GPT and UiO GPT (September / October 2026) 
    Role: The LLM has fixed bugs and written short snippets of code to help with formatting plots.
"""


from src.utils import *

def main(x, y):
    degree = 5
    x_train, _, y_train, _ = train_test_split(x, y, test_size = 0.2, random_state=2026)

    X = rescale_design_matrix(x_train, degree=5)
    methods = ("momentum", "adagrad", "rmsprop", "adam", "plain")
    max_iters = 100
    closed_form_OLS = np.linalg.pinv(X.T @ X) @ X.T @ y_train

    max_gamma_OLS = 2/np.max(hessian_eigs(X, lam=0.0))
    optimal_gamma_OLS = 0.9*max_gamma_OLS

    grads = {'ols': lambda th: gradient(th, x_train, y_train, lam=0.0, degree=degree), 'ridge': lambda th: gradient(th, x_train, y_train, lam=0.01, degree=degree)}

    for method in methods:
        plt.figure(figsize = (9,5))
        for M in [1, 16, 32]:
            for sched in ((1.0, 10.0), (5.0, 50.0), (20.0, 200.0)):
                path_SGD_OLS = stochastic_gradient_descent(X, y_train, method=method, n_epochs=max_iters, batch_size=M, gamma=0.05, schedule=sched, lam=0.0, seed=2026)
                plt.plot(np.arange(len(path_SGD_OLS)), np.linalg.norm(closed_form_OLS - path_SGD_OLS, axis=1), label=f"M={M}, sched={sched}")
        path_GD_OLS = optimise(grads['ols'], np.zeros(X.shape[1]), method=method, gamma=optimal_gamma_OLS, num_iters=max_iters*2, tol=1e-8, lasso_bool = False)
        plt.plot(np.arange(len(path_GD_OLS)), np.linalg.norm(closed_form_OLS - path_GD_OLS, axis=1), label=f"GD")
        # plt.title(f"SGD vs GD of {method} for OLS")
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # standard form
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15
        plt.xlabel("Epochs", fontsize = 15)
        plt.ylabel("Norm of difference to closed-form OLS", fontsize = 15)
        plt.legend(loc="upper right", fontsize=12)
        plt.yscale("log")
        plt.grid(alpha=0.4)
        plt.savefig(f"figures/SGD_OLS/SGD_vs_GD_{method}.pdf", dpi=300, bbox_inches="tight")

    
    lam=0.1
    I = np.eye(X.shape[1])
    n=len(y_train)
    closed_form_ridge = np.linalg.pinv(X.T @ X + n * lam * I) @ X.T @ y_train
    
    max_gamma_ridge = 2/np.max(hessian_eigs(X, lam=lam))
    optimal_gamma_ridge = 0.9*max_gamma_ridge

    grads = {'ols': lambda th: gradient(th, x_train, y_train, lam=0.0, degree=degree), 'ridge': lambda th: gradient(th, x_train, y_train, lam=lam, degree=degree)}

    for method in methods:
        plt.figure(figsize = (9,5))
        for M in [1, 16, 32]:
            for sched in ((1.0, 10.0), (5.0, 50.0), (20.0, 200.0)):
                path_SGD_ridge = stochastic_gradient_descent(X, y_train, method=method, n_epochs=max_iters, batch_size=M, gamma=0.05, schedule=sched, lam=lam, seed=2026)
                plt.plot(np.arange(len(path_SGD_ridge)), np.linalg.norm(closed_form_ridge - path_SGD_ridge, axis=1), label=f"M={M}, sched={sched}")
        path_GD_ridge = optimise(grads['ridge'], np.zeros(X.shape[1]), method=method, gamma=optimal_gamma_ridge, num_iters=max_iters*2, tol=1e-8, lasso_bool = False)
        plt.plot(np.arange(len(path_GD_ridge)), np.linalg.norm(closed_form_ridge - path_GD_ridge, axis=1), label=f"GD")
        # plt.title(f"SGD vs GD of {method} for Ridge")
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # standard form
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15
        plt.xlabel("Epochs", fontsize = 15)
        plt.ylabel("Norm of difference to closed-form Ridge", fontsize = 15)
        plt.legend(loc="upper right", fontsize=12)
        plt.grid(alpha=0.4)
        plt.yscale("log")
        
        plt.savefig(f"figures/SGD_Ridge/SGD_vs_GD_{method}.pdf", dpi=300, bbox_inches="tight")

    