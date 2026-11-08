import os
import time
import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)     # double precision, so that gradient checks agree to ~1e-15
import jax.numpy as jnp
from sklearn.metrics import accuracy_score


SEED = 2026
rng = np.random.default_rng(SEED) #LLM tag: ChatGPT suggested to make this more global, as we were having some issues with reproducibility


# Activation functions

def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))

def ReLU(z):
    return jnp.where(z > 0, z, 0.0)

def leaky_ReLU(z, alpha = 0.01):
    return jnp.where(z > 0, z, alpha*z)

def leaky_ReLU_derivative(z, alpha=0.01):
    return jnp.where(z > 0, 1.0, alpha)

def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data: one row of scores per sample."""
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True)) # LLM-tag (Claude Code) make this True
    return e_z / jnp.sum(e_z, axis=1, keepdims=True)

# hidden activations: (f, f')
ACTIVATIONS = {"sigmoid": (sigmoid, lambda z: sigmoid(z) * (1 - sigmoid(z))),
               "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
               "relu": (ReLU, lambda z: (z > 0).astype(float)),
               "lrelu": (leaky_ReLU, leaky_ReLU_derivative),
               "identity": (lambda z: z, lambda z: np.ones_like(z))}

def create_layers_batch(network_input_size, layer_output_sizes):
    layers = []
    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = rng.normal(0, 1/np.sqrt(i_size), (i_size, layer_output_size))
        b = np.zeros(layer_output_size) + 0.01
        layers.append((W, b))

        i_size = layer_output_size
    return layers

def feed_forward_batch(inputs, layers, hidden_activation="sigmoid", head="cross"):
    """
    LLM tag: Gave ChatGPT some of my functions, as there were some issues with the dictionary used for the activation functions.
    The LLM helped debug some lines. However, the final solution was to hardcode the sigmoid as the activation function, as this is the one
    we wanted to use throughout the code. More debugging was used, solution was to fix string vs activation funciton.
    """
    activation_func = ACTIVATIONS[hidden_activation][0]

    a = inputs
    activations = [inputs]
    zs = []
    for l, (W, b) in enumerate(layers):
        z = a @ W + b
        zs.append(z)

        if l < len(layers) - 1:
            a = activation_func(z)       # hidden layers
        elif head == "cross":
            a = softmax(z)               # multiclass output layer
        elif head == "binary":
            a = sigmoid(z)               # binary output layer
        else:
            a = z                        # linear output for MSE
        activations.append(a)
        
    return zs, activations


def predict(X, layers, hidden_activation="sigmoid", head="cross"):
    return feed_forward_batch(X, layers, hidden_activation, head)[1][-1]


def cost(X, Y, layers, hidden_activation="sigmoid", head="cross", norm="l2", lam=0.0):
    out = predict(X, layers, hidden_activation, head)

    if norm == "l1":
        norm_value = lam * sum(np.sum(np.abs(W)) for W, _ in layers)
    elif norm == "l2":
        norm_value = lam * sum(np.sum(W**2) for W, _ in layers)
    else:
        raise ValueError("norm must be 'l1' or 'l2'")
    p = np.clip(out, 1e-12, 1 - 1e-12)

    if head == "binary":
        p = np.clip(out, 1e-12, 1 - 1e-12)
        data_loss = -np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p))
    elif head == "cross":
        data_loss = -np.mean(np.sum(Y * np.log(p), axis=1))
    else:
        data_loss = 0.5 * np.sum((out - Y) ** 2) / len(X)

    return data_loss + norm_value


def backpropagation(X, Y, layers, hidden_activation="sigmoid", head="cross", norm="l2", lam=0.0):
    # The four equations, for all samples at once. Returns [(dW, db), ...] with the shapes of [(W, b), ...].
    act_function, fp = ACTIVATIONS[hidden_activation]

    z, a = feed_forward_batch(X, layers, hidden_activation, head)
    delta = (a[-1] - Y) / len(X)

    grads = [None] * len(layers)
    for l in range(len(layers) - 1, -1, -1):
        W = layers[l][0]
        dW = a[l].T @ delta
        if norm == "l1":
            dW += lam * np.sign(W)
        elif norm == "l2":
            dW += 2 * lam * W
        else:
            raise ValueError("norm must be 'l1' or 'l2'")
        grads[l] = (dW, np.sum(delta, axis=0))   # dC/dW = A^T Delta,  dC/db = sum of rows
        if l > 0:
            delta = (delta @ W.T) * fp(z[l - 1])         # one layer back
    return grads

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

def train(X, Y, layers, hidden_activation="sigmoid", head="cross", norm="l1", lam=0.0, gamma=0.1, epochs=100, method="plain", rng=None, batch_size=None, X_test=None, y_test=None, tol=1e-6):
    states = [({"W": {}, "b": {}}, {"W": {}, "b": {}}) for W, b in layers]
    history = {"cost": [], "test_accuracy": [], }
    rng = np.random.default_rng(2026) if rng is None else np.random.default_rng(rng)

    n = len(X)
    M = n if batch_size is None else batch_size

    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, M):
            new_layers = []
            new_states = []
            idx = order[s:s + M]
            layers_grad = backpropagation(X[idx], Y[idx], layers, hidden_activation, head, norm, lam)
            for (W, b), (dW, db), (state_W, state_b) in zip(layers, layers_grad, states):
                W, state_W = optimiser_step(method, W, dW, state_W,epoch + 1, gamma)
                b, state_b = optimiser_step(method, b, db, state_b,epoch + 1, gamma)
                new_layers.append((W, b))
                new_states.append((state_W, state_b))
            states = new_states
            layers = new_layers
        history["cost"].append(cost(X, Y, layers, hidden_activation, head, norm, lam))
        if X_test is not None:
            pred = np.argmax(predict(X_test, layers, hidden_activation, head), axis=1)
            history["test_accuracy"].append(np.mean(pred == y_test))

    return layers, history


def one_hot(y, n_classes):
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y

from sklearn.model_selection import StratifiedKFold


def cross_validate_neural_network(X, y, hidden_sizes=[100], method="plain", epochs=50, n_splits=5, norm="l2", lam=0.0, gamma=0.1):
    y = np.array(y)
    folds = StratifiedKFold(n_splits=n_splits,shuffle=True, random_state=2026)

    fold_scores = []

    for fold, (train_indices, val_indices) in enumerate(folds.split(X, y)):
        X_train_fold = X[train_indices]
        y_train_fold = y[train_indices]
        X_val_fold = X[val_indices]
        y_val_fold = y[val_indices]

        rng = np.random.default_rng(2026 + fold)

        layers = create_layers_batch(784, hidden_sizes + [10])

        layers, history = train(
            X_train_fold,
            one_hot(y_train_fold, 10),
            layers,
            hidden_activation="relu",
            head="cross",
            norm=norm,
            lam=lam,
            gamma=gamma,
            method=method,
            batch_size=100,
            epochs=epochs,
            rng=2026 + fold,
            X_test=X_val_fold,
            y_test=y_val_fold
        )

        score = history["test_accuracy"][-1]
        fold_scores.append(score)

        print(f"Fold {fold + 1}: {score:.4f}")

    return np.mean(fold_scores), np.std(fold_scores, ddof=1) / np.sqrt(n_splits)
