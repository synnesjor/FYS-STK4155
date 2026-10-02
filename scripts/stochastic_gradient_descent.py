from src.utils import *

def main(x, y):
    degree = 5
    x_train, _, y_train, _ = train_test_split(x, y, test_size = 0.2, random_state=2026)

    X = rescale_design_matrix(x_train, degree=5)
    n = len(y_train)
    methods = ("plain", "momentum", "adagrad", "rmsprop", "adam")
    iters_OLS = {}
    max_iters = 100
    closed_form_OLS = np.linalg.pinv(X.T @ X) @ X.T @ y_train

    max_gamma = 2/np.max(hessian_eigs(X, lam=0.0))

    grads = {'ols': lambda th: gradient(th, x_train, y_train, lam=0.0, degree=degree), 'ridge': lambda th: gradient(th, x_train, y_train, lam=0.01, degree=degree)}


    for method in methods:
        plt.figure()
        iters_OLS[method] = []
        for M in [1, 32, 64]:
            epoch, path_SGD = stochastic_gradient_descent(X, y_train, method=method, n_epochs=max_iters, batch_size=M, gamma=0.05, schedule=None, lam=0.0, seed=2026)
            if epoch > max_iters:
                iters_OLS[method].append(np.nan)
            else:
                iters_OLS[method].append(epoch+1)
            plt.plot(np.arange(len(path_SGD)), np.linalg.norm(closed_form_OLS - path_SGD, axis=1), label=f"{method}, M={M}")
        path_GD = optimise(grads['ols'], np.zeros(X.shape[1]), method=method, gamma=0.05, num_iters=max_iters*2, tol=1e-8)
        plt.plot(np.arange(len(path_GD)), np.linalg.norm(closed_form_OLS - path_GD, axis=1), label=f"{method}, GD")
        plt.xlabel("Epochs")
        plt.ylabel("Norm of difference to closed-form OLS")
        plt.legend()
        plt.yscale("log")
        plt.grid()


    print(iters_OLS)
