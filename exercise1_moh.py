
def test_gamma(gamma=0.1):
    # Initialize weights for gradient descent
    theta_gdOLS = np.zeros(n_features)
    theta_gdRidge = np.zeros(n_features)

    # Gradient descent loop
    for t in range(num_iters):
        # Compute gradients for OLS and Ridge
        grad_OLS = (2.0 / n) * X_norm.T @ (X_norm @ theta_gdOLS - y_centered)
        # grad_Ridge = (2.0 / n) * X_norm.T @ (X_norm @ theta_gdRidge - y_centered) + 2 * lam * theta_gdRidge
        # Update parameters theta
        theta_gdOLS -= gamma * grad_OLS
        # theta_gdRidge -= gamma * grad_Ridge
        if (np.linalg.norm(grad_OLS)) < 1.0e-8:
            break
    return theta_gdOLS, t