
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

rng = np.random.default_rng()

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
def train_network(inputs, layers, activation_funcs, targets, gamma=0.1, epochs=300):
    history = []
    for epoch in range(epochs):
        layers_grad = gradient_func(inputs, layers, activation_funcs, targets)
        layers = [(W - gamma*dW, b - gamma*db) for (W, b), (dW, db) in zip(layers, layers_grad)]
        if epoch % 50 == 0 or epoch == epochs - 1:
            predictions = feed_forward_batch(inputs, layers, activation_funcs)
            history.append((epoch, float(cost(inputs, layers, activation_funcs, targets)), mse(predictions, targets)))
    return layers, history

inputs = np.random.uniform(-1,1,100).reshape(-1, 1)

network_input_size = inputs.shape[1]
layer_output_sizes = [100, 100, 1]
#activation_funcs = [sigmoid, sigmoid, softmax]
activation_funcs = [sigmoid, sigmoid, no_activation_func]

layers_batch = create_layers_batch(network_input_size, layer_output_sizes)

predictions = feed_forward_batch(inputs, layers_batch, activation_funcs)

targets = runge_1d(inputs)
print(targets)

gradient_func = jax.grad(cost, argnums=1)  # Taking the gradient wrt. the second input to the cost function, i.e. the layers
layers_grad = gradient_func(inputs, layers_batch, activation_funcs, targets) 

trained, history = train_network(inputs, layers_batch, activation_funcs, targets)
for epoch, c, mse in history:
    print(f"epoch {epoch:4d}: cost {c:.4f}  mse {mse:.4f}")


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

#X = rescale_design_matrix(inputs,deg=6)
mse_train, mse_test, theta,  y_tilde, y_tilde_test, y_tilde_train, y_train, y_test = calculate_mse(inputs, targets, degree=6)
