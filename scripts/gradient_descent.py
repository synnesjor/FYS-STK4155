import numpy as np
from sklearn.linear_model import Ridge
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad
from src.utils import *


def main(x, y, n, rng):
    # Set regularization parameter, either a single value or a vector of values 
    # Note that lambda is a python keyword: the lambda keyword creates small anonymous functions. # 2/n * X.T @ X + 2 * lam * I
    y = y - y.mean() # centre y

    lam = 0.1
    gamma = 0.1

    def grad_ridge(theta, gamma, X, y, lam, max_iter=10000, tol=1e-8): # works for OLS with lam=0.0
        for k in range(max_iter):
            gradient = (2.0 / n) * X.T @ (X @ theta- y) + 2 * lam * theta
            theta -= gamma * gradient
            if np.linalg.norm(gradient) < tol:
                break
        return theta, k


    for deg in [2,5,15]:

        X_norm = rescale_design_matrix(x, deg)
        y_centered = y - y.mean()    

        # Analytical forms: theta_Ridge = (X^T X + n*lambda*I)^{-1} X^T y and theta_OLS = (X^T X)^{-1} X^T y
        n_features = X_norm.shape[1]
        theta = rng.normal(size = n_features)
        I = np.eye(n_features)
        theta_closed_formRidge = np.linalg.pinv(X_norm.T @ X_norm + n * lam * I) @ X_norm.T @ y_centered
        theta_closed_formOLS = np.linalg.pinv(X_norm.T @ X_norm) @ X_norm.T @ y_centered


        theta_OLS, k_OLS = grad_ridge(theta, gamma, X_norm, y_centered, lam=0.0) # OLS
        print(f"Number of iterations for deg={deg} is (OLS):", k_OLS)

        theta_ridge, k_ridge = grad_ridge(theta, gamma, X_norm, y_centered, lam) # Ridge
        print(f"Number of iterations for deg={deg} is (Ridge):", k_ridge)

        print(f"Gradient descent after {k_ridge+1} iterations (Ridge):", theta_ridge.ravel())
        print(f"Gradient descent after {k_OLS+1} iterations (OLS):", theta_OLS.ravel())

        print("Closed-form Ridge coefficients:", theta_closed_formRidge)
        print("Closed-form OLS coefficients:", theta_closed_formOLS)

        clf = Ridge(alpha = lam*n, fit_intercept=False)
        clf.fit(X_norm, y_centered)

        print(f"Coefficients from scikit-learn Ridge():", clf.coef_)

        # second part

        lam = 1e-2
        theta0 = rng.normal(size=X_norm.shape[1])

        def cost_ols(theta, X, y):
            return jnp.mean((y - X @ theta)**2)

        def cost_ridge(theta, X, y, lam):
            return jnp.mean((y - X @ theta)**2) + lam * jnp.sum(theta**2)

        # gradients by automatic differentiation (with respect to argument 0 = theta)
        grad_ols_ad = grad(cost_ols)
        grad_ridge_ad = grad(cost_ridge)

        # the same gradients by hand
        grad_ols_analytic = 2.0 / len(y_centered) * X_norm.T @ (X_norm @ theta0 - y_centered)
        grad_ridge_analytic = grad_ols_analytic + 2.0 * lam * theta0

        print(f"OLS   max |AD - analytic| for deg={deg}", np.max(np.abs(grad_ols_ad(theta0, X_norm, y_centered) - grad_ols_analytic)))
        print(f"Ridge max |AD - analytic| for deg={deg}", np.max(np.abs(grad_ridge_ad(theta0, X_norm, y_centered, lam) - grad_ridge_analytic)))

        # the largest safe learning rate for plain gradient descent: eta < 2 / lambda_max(Hessian)
        H = 2.0 / len(y) * X_norm.T @ X_norm
        print("eta_max for OLS  =", 2.0 / np.linalg.eigvalsh(H).max())