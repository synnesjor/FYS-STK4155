import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
jax.config.update("jax_enable_x64", True)
from pathlib import Path

Path("../data").mkdir(exist_ok=True)

SEED = 2026
rng = np.random.default_rng(SEED) #LLM tag: ChatGPT suggested to make this more global, as we were having some issues with reproducibility


def predict_network(X, layers, activation_functions):
    return feed_forward_batch(X, layers, activation_functions)


def runge_1d(x, sigma, n):
    noise = rng.normal(0, sigma, n).reshape(-1, 1)
    return (1 / (1+(25*x**2))) + noise

def runge_1d_no_noise(x):
    return (1 / (1+(25*x**2)))


def no_activation_func(z):
    return z


def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))


def ReLU(z):
    return jnp.where(z > 0, z, 0.0)


def leaky_ReLU(z, alpha = 0.01):
    return jnp.where(z > 0, z, alpha*z)


def softmax(z):
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=False))
    return e_z / jnp.sum(e_z, axis=1, keepdims=True)


def softmax_vec(z):
    e_z = jnp.exp(z - jnp.max(z))
    return e_z / jnp.sum(e_z)


def weights(n_in, n_out):
    return np.random.normal(0, 2 / (n_in + n_out))  #her vil vi ha n_in = n_out fra forrige layer, n_out er definert av oss i layer_output_size


def create_layers_batch(network_input_size, layer_output_sizes):
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = rng.normal(0, 1/np.sqrt(i_size), (i_size, layer_output_size))
        # print(np.shape(W))
        b = np.zeros(layer_output_size) + 0.01
        layers.append((W, b))

        i_size = layer_output_size
    return layers


def mse(predict, target):
    return jnp.mean((target-predict)**2)


def cost(inputs, layers, target, hidden_activation="sigmoid", norm="l2", lam=0.0):
    out = predict_network(inputs, layers, hidden_activation)

    if norm == "l1":
        norm_value = lam * sum(jnp.sum(jnp.abs(W)) for W, _ in layers)
    elif norm == "l2":
        norm_value = lam * sum(jnp.sum(W**2) for W, _ in layers)
    else:
        raise ValueError("norm must be 'l1' or 'l2'")

    data_loss = mse(out, target)
    return data_loss + norm_value


def feed_forward_batch(inputs, layers, activation_funcs):
    a = inputs
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = a @ W + b
        a = activation_func(z)
    return a

def accuracy(predictions, targets):
    return accuracy_score(np.argmax(np.asarray(predictions), axis=1), np.argmax(np.asarray(targets)))

def gradient_func(inputs, layers, activation_functions, targets, norm="l2", lam=0.0):
    '''
    LLM tag: Claude code, how to write code in needed form for JAX (using lambda command)
    '''
    cost_value = lambda layer_params: cost(
        inputs, layer_params, targets, activation_functions, norm, lam
    )
    return jax.grad(cost_value)(layers)


# i b så ber de om å bruke bare mse - accuracy er ett klassifikasjon problem og vi gjør regression her
def train_network(inputs, layers, activation_funcs, targets, gamma=0.1, epochs = 10000):
    history = []
    for epoch in range(epochs):
        layers_grad = gradient_func(inputs, layers, activation_funcs, targets)
        layers = [(W - gamma*dW, b - gamma*db) for (W, b), (dW, db) in zip(layers, layers_grad)]
        if epoch % 50 == 0 or epoch == epochs - 1:
            predictions = feed_forward_batch(inputs, layers, activation_funcs)
            history.append((epoch, float(cost(inputs, layers, targets, activation_funcs)), mse(predictions, targets)))
    return layers, history, predictions, mse(predictions, targets)

def train_network_gradient(inputs, layers, activation_funcs, targets, norm="l2", lam=0.0, gamma=0.01, epochs=10000, method="plain", tol = 0):
    """
    LLM tag:
    With the train_network function as a base, we asked ChatGPT and Claude if it made sense to reuse optimiser step function used in Project 1. 
    When we had our own suggestion for how to combine the functions, we checked with the LLMs if they agreed. We also got the line creating the 
    dictionary for "states", and got some help with debbugging the loop where the layer's weight and biases are updated.     
    """
    # Independent optimizer states for each weight and bias
    states = [({"W": {}, "b": {}}, {"W": {}, "b": {}}) for W, b in layers]
    history = []
    for epoch in range(epochs):
        # param_old = param.copy()
        # Calculate gradients for the whole network
        layers_grad = gradient_func(
            inputs, layers, activation_funcs, targets, norm=norm, lam=lam
        )

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
            c = cost(inputs, layers, targets, activation_funcs, norm=norm, lam=lam)
            # acc = accuracy(predictions, targets)
            history.append((epoch, float(c), mse(predictions, targets)))

            # if np.abs(c[epoch]-c[epoch-1]) < tol: #this needs to be changed into a proper tolerance definition
            #     print(f"Converged at epoch {epoch}: cost = {c:.6g}")
            #     break
        
        # if np.linalg.norm(param - param_old)>1e2: # breaks if param explodes
        #     break

    return layers, history, predictions, mse(predictions,targets)

def step_length(t, t0, t1):
    return t0 / (t + t1)

def train_network_stochastic_gradient(inputs, layers, activation_funcs, targets, method, norm="l2", lam=0.0, schedule = None, gamma=0.01, epochs=10000, batch_size=5, tol=1e-6):
    """
    LLM tag:
    With the train_network function as a base, we asked ChatGPT and Claude if it made sense to reuse optimiser step function used in Project 1.
    When we had our own suggestion for how to combine the functions, we checked with the LLMs if they agreed, especially with the 
    placement of the epochs loop, and updates of t. We also got the convergence check from ChatGPT.
    """
    # Independent optimizer states for each weight and bias
    states = [({"W": {}, "b": {}}, {"W": {}, "b": {}}) for W, b in layers]
    history = []
    t = 0
    for epoch in range(epochs):
        # Calculate gradients for the whole network
        for batch in make_batches(len(inputs), batch_size, rng):
            t += 1
            layers_grad = gradient_func(inputs[batch], layers, activation_funcs, targets[batch])
            gamma_t = gamma if schedule is None else step_length(t, *schedule)    
        
            new_layers = []
            new_states = []

            # Update each layer's weights and biases
            for (W, b), (dW, db), (state_W, state_b) in zip(layers, layers_grad, states):
                W, state_W = optimiser_step(method, W, dW, state_W,epoch + 1, gamma_t)
                b, state_b = optimiser_step(method, b, db, state_b,epoch + 1, gamma_t)
                new_layers.append((W, b))
                new_states.append((state_W, state_b))
            layers = new_layers
            states = new_states

        # Record training progress
        if epoch % 50 == 0 or epoch == epochs - 1: #find out what this does
            predictions = feed_forward_batch(inputs, layers, activation_funcs)
            c = cost(inputs, layers, targets, activation_funcs, norm=norm, lam=lam)
            # acc = accuracy(predictions, targets)
            history.append((epoch, float(c), mse(predictions, targets)))

            if c < tol:
                print(f"Converged at epoch {epoch}: cost = {c:.6g}")
                # break


    return layers, history, predictions, mse(predictions,targets)


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

def gradient_batch(theta, X, y, lam=0.0):
    n = len(y)
    return (2.0 / n) * X.T @ (X @ theta - y) + 2.0 * lam * theta

def make_batches(n, batch_size, rng):
    idx = rng.permutation(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]

def step_length(t, t0, t1):
    return t0 / (t + t1)

#for OLS comparison
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
    y = y - np.mean(y)
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
