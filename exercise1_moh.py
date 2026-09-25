

# Set regularization parameter, either a single value or a vector of values 
# Note that lambda is a python keyword: the lambda keyword creates small anonymous functions. # 2/n * X.T @ X + 2 * lam * I
lam = 0.1

# Analytical forms: theta_Ridge = (X^T X + n*lambda*I)^{-1} X^T y and theta_OLS = (X^T X)^{-1} X^T y
n_features = X_norm.shape[1]
I = np.eye(n_features)
theta_closed_formRidge = np.linalg.pinv(X_norm.T @ X_norm + n * lam * I) @ X_norm.T @ y_centered
theta_closed_formOLS = np.linalg.pinv(X_norm.T @ X_norm) @ X_norm.T @ y_centered

print("Closed-form Ridge coefficients:", theta_closed_formRidge)
print("Closed-form OLS coefficients:", theta_closed_formOLS)

theta = rng.normal(size=n_features)
gamma=0.1

def grad_ridge(theta, gamma, X, y, lam):
    for k in range(1000):
        gradient = (2.0 / n) * X.T @ (X @ theta- y) + 2 * lam * theta
        theta_new = theta - gamma * gradient
        theta-= gamma * gradient
        if np.linalg.norm(gradient) < 1.0e-8:
            break
    return theta

def grad_OLS(theta, gamma, X, y):
    for k in range(1000):
        gradient = (2.0 / n) * X.T @ (X @ theta- y)
        theta-= gamma * gradient
        if np.linalg.norm(gradient) < 1.0e-8:
            break
    return theta    

theta_ridge=grad_ridge(theta, gamma, X_norm, y_centered, lam)
theta_OLS=grad_OLS(theta, gamma, X_norm, y_centered)

print(f"Gradient descent after {k+1} iterations (Ridge):", theta_ridge.ravel())
print(f"Gradient descent after {k+1} iterations (OLS):", theta_OLS.ravel())

from sklearn.linear_model import Ridge

clf = Ridge(alpha=lam*n, fit_intercept=False)
clf.fit(X_norm, y_centered)

print(f"Coefficients from scikit-learn Ridge():", clf.coef_)

# second part

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad

def cost_OLS(theta, X, y):
    return 1/n * jnp.linalg.norm(X@theta-y, ord=2)

def cost_Ridge(theta, X, y, lam):
    return 1/n * jnp.linalg.norm(X@theta-y, ord=2) + lam * theta.T @ theta

theta_test = rng.standard_normal(n_features)
grad_OLS_ad = grad(cost_OLS)(theta_test, jnp.asarray(X_norm), jnp.asarray(y_centered))
grad_OLS_analytical = (2.0 / n) * X_norm.T @ (X_norm @ theta_gdOLS - y_centered)
print("max |AD - analytical| =", np.max(np.abs(np.asarray(grad_OLS_ad) - grad_OLS_analytical)))