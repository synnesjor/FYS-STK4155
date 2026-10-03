import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import jax
import jax.numpy as jnp

# Useful Functions

def design_matrix(x, degree, intercept = False):
    start = 0 if intercept else 1
    return np.vstack([x**p for p in range(start, degree + 1)]).T

def rescale_design_matrix(x, degree):
    X = design_matrix(x, degree)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1  # safeguard to avoid division by zero for constant features
    X_norm = (X - X_mean) / X_std
    return X_norm

def ols(X, y):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ ((U.T @ y) / s)

def calculate_mse(x, y, degree):
    X = rescale_design_matrix(x, degree)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 2026)
    model = LinearRegression(fit_intercept=False)
    model.fit(X_train, y_train)
    y_tilde = model.predict(X)
    y_tilde_test = model.predict(X_test)
    y_tilde_train = model.predict(X_train)
    mean_squared_error_train = mean_squared_error(y_train, model.predict(X_train))
    mean_squared_error_test = mean_squared_error(y_test, model.predict(X_test))
    theta = ols(X_test, y_test)
    return mean_squared_error_train, mean_squared_error_test, theta, y_tilde, y_tilde_test, y_tilde_train, y_train, y_test


def ridge(x,y,lam,degree):
    X = rescale_design_matrix(x, degree)
    return ols(X,y) * (1/(1+lam))

def grad_ridge(theta, gamma, X, y, lam, max_iter=100000, tol=1e-6): # works for OLS with lam=0.0
    n = len(y)
    for k in range(max_iter):
        gradient = (2.0 / n) * X.T @ (X @ theta- y) + 2 * lam * theta
        theta -= gamma * gradient
        if np.linalg.norm(gradient) < tol:
            print("converges at", k)
            break
    return theta, k

def gradient(theta, x, y, degree, lam=0.0):
    """Eqs. (4.13) and (4.17): the gradient of (1/n)||X theta - y||^2 + lambda theta^T theta."""
    X = rescale_design_matrix(x, degree)
    n = len(y)
    return (2.0 / n) * X.T @ (X @ theta - y) + 2.0 * lam * theta

def gradient_batch(theta, X, y, lam=0.0):
    """Eqs. (4.13) and (4.17): the gradient of (1/n)||X theta - y||^2 + lambda theta^T theta."""
    n = len(y)
    return (2.0 / n) * X.T @ (X @ theta - y) + 2.0 * lam * theta

def gradient_lasso(theta, x, y, degree, lam=0.0):
    X = rescale_design_matrix(x, degree)
    n = len(y)

    residual = y - X @ theta

    mse_grad = -(2 / n) * X.T @ residual
    l1_grad = lam * np.sign(theta)

    return mse_grad + l1_grad


def hessian_eigs(X, lam):
    """Eigenvalues of the Hessian (2/n) X^T X + 2 lambda I, Eqs. (4.14) and (4.17)."""
    n = len(X)
    return np.linalg.eigvalsh((2.0 / n) * X.T @ X + 2.0 * lam * np.eye(X.shape[1]))

def optimiser_step(method, theta, g, state, t, gamma, beta=0.9, rho=0.99,
                beta1=0.9, beta2=0.999, eps=1e-6):
    """One update of theta from the gradient g at step t (t = 1, 2, ...), Eqs. (4.10), (4.28),
    (4.42)-(4.45), (4.47)-(4.48) and (4.51)-(4.55).  state carries the running quantities."""
    if method == "plain":
        return theta - gamma * g, state
    if method == "momentum":
        v = beta * state.get("v", 0.0) + gamma * g               # Eq. (4.28)
        state["v"] = v
        return theta - v, state
    if method == "adagrad":
        r = state.get("r", 0.0) + g * g                           # Eq. (4.42)
        state["r"] = r
        return theta - gamma * g / (np.sqrt(r) + eps), state      # Eq. (4.45)
    if method == "rmsprop":
        r = rho * state.get("r", 0.0) + (1.0 - rho) * g * g       # Eq. (4.47)
        state["r"] = r
        return theta - gamma * g / (np.sqrt(r) + eps), state      # Eq. (4.48)
    if method == "adam":
        m = beta1 * state.get("m", 0.0) + (1.0 - beta1) * g       # Eq. (4.51)
        r = beta2 * state.get("r", 0.0) + (1.0 - beta2) * g * g   # Eq. (4.52)
        state["m"], state["r"] = m, r
        m_hat = m / (1.0 - beta1**t)                              # Eq. (4.54)
        r_hat = r / (1.0 - beta2**t)
        return theta - gamma * m_hat / (np.sqrt(r_hat) + eps), state   # Eq. (4.55)
    raise ValueError(f"unknown method {method}")

def optimise(grad, theta0, method, gamma, num_iters = 100, tol = 1e-6, **kw):
    """Run one optimiser from theta0 with the full gradient; returns all iterates."""
    theta, state = np.array(theta0, dtype=float), {}
    history = [theta.copy()]
    for t in range(1, num_iters + 1):
        g = grad(theta)
        theta_old = theta.copy()
        theta, state = optimiser_step(method, theta, g, state, t, gamma, **kw)
        history.append(theta.copy())
        # if np.linalg.norm(g) < tol:
        #     # print("Final iteration", t)
        #     # print("Final theta", theta)
        #     break
        if np.linalg.norm(theta - theta_old) < tol:
            break
    return history

def make_batches(n, batch_size, rng):
    """Shuffle the indices and split them into minibatches."""
    idx = rng.permutation(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]

def step_length(t, t0, t1):
    """The schedule of Eq. (4.40)."""
    return t0 / (t + t1)

def stochastic_gradient_descent(X, y, method="plain", n_epochs=50, batch_size=5, gamma=0.1, schedule=None, lam=0.0, seed=2026, tol=1e-6, **kw,):
    """Minibatch SGD, Eq. (4.34), with any optimiser. schedule=(t0, t1) replaces gamma by Eq. (4.40).
    Returns the iterate after every epoch."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    theta, state, t = np.zeros(p), {}, 0
    history = [theta.copy()]
    
    for epoch in range(n_epochs):
        theta_old = theta.copy()
        for batch in make_batches(n, batch_size, rng):
            t += 1
            g = gradient_batch(theta, X[batch], y[batch], lam)                                    # the gradient of the cost on this minibatch
            gamma_t = gamma if schedule is None else step_length(t, *schedule)                           # constant, or the schedule
            theta, state = optimiser_step(method, theta, g, state, t, gamma_t, **kw)
        if np.linalg.norm(theta - theta_old) < tol:
            break
        if np.linalg.norm(theta - theta_old)>1e2: # breaks if theta explodes
            break
        history.append(theta.copy())
    return np.array(history)