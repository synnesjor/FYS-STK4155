
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


rng = np.random.default_rng(seed = 2026)
np.random.seed = 2026

def runge_1d(x):
    return 1 / (1+(25*x**2))

def no_activation_func(z):
    return z

def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))

def ReLU(z):
    return jnp.where(z > 0, z, 0.0)

def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data: one row of scores per sample."""
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=False))
    return e_z / jnp.sum(e_z, axis=1, keepdims=True)


def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.
    Use this function when you use the activation function on one vector at a time."""
    e_z = jnp.exp(z - jnp.max(z))
    return e_z / jnp.sum(e_z)


def weights(n_in, n_out):
    return np.random.normal(0, 2 / (n_in + n_out))  #her vil vi ha n_in = n_out fra forrige layer, n_out er definert av oss i layer_output_size

def create_layers_batch(network_input_size, layer_output_sizes):
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = rng.normal(0, 1/np.sqrt(i_size), (i_size, layer_output_size))
        print(np.shape(W))
        b = np.zeros(layer_output_size) + 0.01
        layers.append((W, b))

        i_size = layer_output_size
    return layers


def mse(predict, target):
    return jnp.mean((target-predict)**2)


def cost(input, layers, activation_funcs, target):
    predict = feed_forward_batch(input, layers, activation_funcs)
    return mse(predict, target)


def feed_forward_batch(inputs, layers, activation_funcs):
    a = inputs
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = a @ W + b
        a = activation_func(z)
    return a

def accuracy(predictions, targets):
    """Fraction of rows whose largest predicted probability sits in the correct class."""
    return accuracy_score(np.argmax(np.asarray(predictions), axis=1), np.argmax(np.asarray(targets)))

# i b så ber de om å bruke bare mse - accuracy er ett klassifikasjon problem og vi gjør regression her
def train_network(inputs, layers, activation_funcs, targets, gamma=0.1, epochs=10000):
    history = []
    for epoch in range(epochs):
        layers_grad = gradient_func(inputs, layers, activation_funcs, targets)
        layers = [(W - gamma*dW, b - gamma*db) for (W, b), (dW, db) in zip(layers, layers_grad)]
        if epoch % 50 == 0 or epoch == epochs - 1:
            predictions = feed_forward_batch(inputs, layers, activation_funcs)
            history.append((epoch, float(cost(inputs, layers, activation_funcs, targets)), mse(predictions, targets)))
    return layers, history, predictions

def train_network_gradient(inputs, layers, activation_funcs, targets, gamma=0.01, epochs=10000, method="plain"):
    """
        LLM tag needed here
    """
    # Independent optimizer states for each weight and bias
    states = [({"W": {}, "b": {}}, {"W": {}, "b": {}}) for W, b in layers]
    history = []
    for epoch in range(epochs):
        # Calculate gradients for the whole network
        layers_grad = gradient_func(inputs, layers, activation_funcs, targets)

        new_layers = []
        new_states = []

        # Update each layer's weights and biases
        for (W, b), (dW, db), (state_W, state_b) in zip(layers, layers_grad, states):
            W, state_W = optimiser_step(method, W, dW, state_W,epoch + 1, gamma)
            b, state_b = optimiser_step(method, b, db, state_b,epoch + 1, gamma)
            new_layers.append((W, b))
            new_states.append((state_W, state_b))
        layers = new_layers
        states = new_states

        # Record training progress
        if epoch % 50 == 0 or epoch == epochs - 1:
            predictions = feed_forward_batch(inputs, layers, activation_funcs)
            c = cost(inputs, layers, activation_funcs, targets)
            # acc = accuracy(predictions, targets)
            history.append((epoch, float(c), mse(predictions, targets)))

    return layers, history, predictions


def inputs_scale(x):
    X = x
    X_mean = X.mean()
    X_std = X.std()
    # X_std[X_std == 0] = 1  # safeguard to avoid division by zero for constant features
    X_norm = (X - X_mean) / X_std
    return X_norm


def optimiser_step(method, param, g, state, t, gamma, beta=0.9, rho=0.99,
                beta1=0.9, beta2=0.999, eps=1e-6):
    """One update of param from the gradient g at step t (t = 1, 2, ...), Eqs. (4.10), (4.28),
    (4.42)-(4.45), (4.47)-(4.48) and (4.51)-(4.55).  state carries the running quantities."""
    if method == "plain":
        return param - gamma * g, state
    if method == "momentum":
        v = beta * state.get("v", 0.0) + gamma * g               # Eq. (4.28)
        state["v"] = v
        return param - v, state
    if method == "adagrad":
        r = state.get("r", 0.0) + g * g                           # Eq. (4.42)
        state["r"] = r
        return param - gamma * g / (np.sqrt(r) + eps), state      # Eq. (4.45)
    if method == "rmsprop":
        r = rho * state.get("r", 0.0) + (1.0 - rho) * g * g       # Eq. (4.47)
        state["r"] = r
        return param - gamma * g / (np.sqrt(r) + eps), state      # Eq. (4.48)
    if method == "adam":
        m = beta1 * state.get("m", 0.0) + (1.0 - beta1) * g       # Eq. (4.51)
        r = beta2 * state.get("r", 0.0) + (1.0 - beta2) * g * g   # Eq. (4.52)
        state["m"], state["r"] = m, r
        m_hat = m / (1.0 - beta1**t)                              # Eq. (4.54)
        r_hat = r / (1.0 - beta2**t)
        return param - gamma * m_hat / (np.sqrt(r_hat) + eps), state   # Eq. (4.55)
    raise ValueError(f"unknown method {method}")

def optimise(grad, theta0, method, gamma, num_iters = 100, tol = 1e-6, lasso_bool=False, **kw):
    """Run one optimiser from theta0 with the full gradient; returns all iterates."""
    theta, state = np.array(theta0, dtype=float), {}
    history = [theta.copy()]
    for t in range(1, num_iters + 1):
        g = grad(theta)
        theta_old = theta.copy()
        theta, state = optimiser_step(method, theta, g, state, t, gamma, **kw)
        history.append(theta.copy())
        if lasso_bool:
            # Check for convergence based on the change in theta
            if np.linalg.norm(theta - theta_old) < tol:
                break
        elif np.linalg.norm(g) < tol:
            break
    history = np.array(history)
    return history


inputs = np.random.uniform(-1,1,100).reshape(-1, 1)
inputs_scaled = inputs_scale(inputs)


targets = runge_1d(inputs)
targets_centered = targets - np.mean(targets)

gradient_func = jax.grad(cost, argnums=1)  # Taking the gradient wrt. the second input to the cost function, i.e. the layers




#for two hidden layers with size 100
network_input_size = inputs_scaled.shape[1]
layer_output_sizes_100 = [100, 100, 1]
activation_funcs_100 = [sigmoid, sigmoid, no_activation_func]
layers_batch_100 = create_layers_batch(network_input_size, layer_output_sizes_100)
predictions_100 = feed_forward_batch(inputs_scaled, layers_batch_100, activation_funcs_100)
layers_grad_100 = gradient_func(inputs_scaled, layers_batch_100, activation_funcs_100, targets_centered)
# trained_100, history_100, preds_100 = train_network(inputs_scaled, layers_batch_100, activation_funcs_100, targets_centered)

colours = ["#aa333c", "#99d2fb", "#9aab64", "#21534c", "#ef9d2c", "#B0a6df"]
learning_rates = [0.001, 0.01, 0.1, 0.3]
plt.figure()
for idx, each in enumerate(learning_rates):
    trained_plain, history_plain, predictions_plain = train_network_gradient(inputs_scaled, layers_batch_100, activation_funcs_100, targets_centered, gamma = each, epochs=10000, method="plain")
    trained_plain_adam, history_plain_adam, predictions_plain_adam = train_network_gradient(inputs_scaled, layers_batch_100, activation_funcs_100, targets_centered, gamma = each, epochs=10000, method="adam")
    plt.plot(inputs_scaled, predictions_plain, color = colours[idx], label = "plain")
    plt.plot(inputs_scaled, predictions_plain_adam, color = colours[idx+3], label = "adam")
plt.plot(inputs_scaled, targets_centered,  "o", color = "#3ba336", label = "True function", markersize = 2)
plt.show()
# print(predictions_plain)
# for epoch, c, mse in history_plain:
#     print(f"epoch {epoch:4d}: cost {c:.4f}  mse {mse:.4f}")




# #for one hidden layer with size 50
# layer_output_sizes_50 = [50,1]
# activation_funcs_50 = [sigmoid, no_activation_func]
# layers_batch_50 = create_layers_batch(network_input_size, layer_output_sizes_50)
# predictions = feed_forward_batch(inputs_scaled, layers_batch_50, activation_funcs_50)
# layers_grad_50 = gradient_func(inputs_scaled, layers_batch_50, activation_funcs_50, targets_centered) 
# trained_50, history_50, preds_50 = train_network(inputs_scaled, layers_batch_50, activation_funcs_50, targets_centered)
# # for epoch, c, mse in history:
# #     print(f"epoch {epoch:4d}: cost {c:.4f}  mse {mse:.4f}")


"""
Comparison to OLS regression values.
"""

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

# print(np.shape(targets_centered))
# print(np.shape(inputs))

#X = rescale_design_matrix(inputs,deg=6)
mse_train, mse_test, theta, y_tilde, y_tilde_test, y_tilde_train, y_train, y_test = calculate_mse(inputs_scaled[:,0], targets_centered[:,0], degree=6)

# print(preds)

plt.figure()
# plt.plot(inputs_scaled, preds_100, "o", color = "#ff46a2", label = "Neural network size 100", markersize = 2)
# plt.plot(inputs_scaled, preds_50, "o", color = "#8f1cab", label = "Neural network size 50", markersize = 2)
plt.plot(inputs_scaled, predictions_plain, "o", color = "#e5194f", label = "Neural network plain gradient decent layer size 100", markersize = 2)
plt.plot(inputs_scaled, y_tilde, "o", color = "#2da9cf", label = "Polyomial fit", markersize = 2)
plt.plot(inputs_scaled, targets_centered,  "o", color = "#3ba336", label = "True function", markersize = 2)
plt.legend()
plt.show()

