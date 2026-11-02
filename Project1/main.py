import numpy as np
import matplotlib.pyplot as plt

from scripts import test_ols, ridge_regression, bias_variance, cross_validation
from scripts import gradient_descent, optimisers, lasso, stochastic_gradient_descent
from scripts import model_selection_ridge, model_selection_ols, model_selection_lasso
from scripts import plot_estimated_runge


seed = np.random.seed(2026)

def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

def main():
    # Initialise values
    rng = np.random.default_rng(2026)
    n = 100 # also choose 250
    sigma = 0.1 # also choose 0.01
    x = np.sort(rng.uniform(-1, 1, n))
    y = runge(x) + rng.normal(0, sigma, n)

    # Center the target to zero mean
    y_offset = y.mean()
    y = y - y_offset

    # Scripts need certain initial values to run

    # test_ols.main(x, y) # runs part 1a
    # ridge_regression.main(x, y) # runs part 1b
    # bias_variance.main(x, y) # runs part 1c
    # cross_validation.main(x, y) # runs part 1d
    # gradient_descent.main(x, y, n, rng) # runs part 1e
    # optimisers.main(x, y) # runs part 1f
    lasso.main(x, y) # runs part 1g
    # stochastic_gradient_descent.main(x, y) # runs part 1h
    
    # Scripts for part 1i

    # model_selection_ols.main(x, y) 
    # model_selection_ridge.main(x, y) 
    # model_selection_lasso.main(x, y) 
    # plot_estimated_runge.main(x, y, y_offset=y_offset)

    # plt.show()
if __name__ == "__main__":
    main()