
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

rng = np.random.default_rng()


def runge_1d(x):
    return 1 / (1+(25*x**2))


def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))

def ReLU(z):
    return jnp.where(z > 0, z, 0.0)

def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data: one row of scores per sample."""
    e_z = jnp.exp(z - jnp.max(z, axis=0, keepdims=False))
    return e_z / jnp.sum(e_z, axis=0, keepdims=True)


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
    return accuracy_score(np.argmax(np.asarray(predictions), axis=0), np.argmax(np.asarray(targets)))


inputs = np.random.uniform(-1,1,100)

network_input_size = len(inputs)
layer_output_sizes = [12, 10, 1]
activation_funcs = [sigmoid, sigmoid, softmax]

layers_batch = create_layers_batch(network_input_size, layer_output_sizes)


# feed_forward_batch(inputs, layers_batch, activation_funcs)


predictions = feed_forward_batch(inputs, layers_batch, activation_funcs)

targets = runge_1d(inputs)
print(targets)

gradient_func = jax.grad(cost, argnums=1)  # Taking the gradient wrt. the second input to the cost function, i.e. the layers
layers_grad = gradient_func(inputs, layers_batch, activation_funcs, targets) 

def train_network(inputs, layers, activation_funcs, targets, gamma=1.0, epochs=300):
    history = []
    for epoch in range(epochs):
        layers_grad = gradient_func(inputs, layers, activation_funcs, targets)
        layers = [(W - gamma*dW, b - gamma*db) for (W, b), (dW, db) in zip(layers, layers_grad)]
        if epoch % 50 == 0 or epoch == epochs - 1:
            predictions = feed_forward_batch(inputs, layers, activation_funcs)
            history.append((epoch, float(cost(inputs, layers, activation_funcs, targets)), accuracy(predictions, targets)))
    return layers, history

trained, history = train_network(inputs, layers_batch, activation_funcs, targets)
for epoch, c, acc in history:
    print(f"epoch {epoch:4d}: cost {c:.4f}  accuracy {acc:.4f}")



