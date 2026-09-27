import numpy as np
import matplotlib.pyplot as plt

from scripts import test_ols, ridge_regression, bias_variance, cross_validation, gradient_descent, optimisers

def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

def main():
    # Initialise values
    rng = np.random.default_rng(2026)
    n = 100
    sigma = 0.1                                  # noise level: explore it!
    x = np.sort(rng.uniform(-1, 1, n))
    y = runge(x) + rng.normal(0, sigma, n)

    # Scripts need certain initial values to run

    # test_ols.main(x, y) # runs part 1a
    # ridge_regression.main(x, y) # runs part 1b
    # bias_variance.main(x, y) # runs part 1c
    # cross_validation.main(x, y) # runs part 1d
    gradient_descent.main(x, y, n, rng) # runs part 1e
    # optimisers.main(x, y) # runs part 1f

if __name__ == "__main__":
    main()