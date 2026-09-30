from src.utils import *

def main(x, y):

    x_train, _, y_train, _ = train_test_split(x, y, test_size = 0.2, random_state=2026)

    X = rescale_design_matrix(x_train, degree=5)
    n = len(y_train)
    methods = ("plain", "momentum", "adagrad", "rmsprop", "adam")
    iters_OLS = {}
    max_iters = 10000
    gammas = np.logspace(-4,0,10)
    for method in methods:
        iters_OLS[method] = []
        for gam in gammas:
            epoch, path = stochastic_gradient_descent(X, y, method=method, n_epochs=max_iters, batch_size=5, gamma=gam, schedule=None, lam=0.0, seed=2026)
            if epoch > max_iters:
                iters_OLS[method].append(np.nan)
            else:
                iters_OLS[method].append(epoch+1)
    print(iters_OLS)