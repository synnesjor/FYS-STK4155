import os
import time
import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)     # double precision, so that gradient checks agree to ~1e-15
import jax.numpy as jnp

rng = np.random.default_rng(2026)

np.random.default_rng(2026)

# your Project 2 network (or the week-41 code: create_layers, feed_forward, backpropagation, train, predict)

def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))


def ReLU(z):
    return jnp.where(z > 0, z, 0.0)

def leaky_ReLU(z, alpha = 0.01):
    return jnp.where(z > 0, z, alpha*z)

def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data: one row of scores per sample."""
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=False))
    return e_z / jnp.sum(e_z, axis=1, keepdims=True)

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

def feed_forward_batch(inputs, layers, activation_funcs):
    """
    LLM tag: Gave ChatGPT some of my functions, as there were some issues with the dictionary used for the activation functions.
    The LLM helped debug some lines. However, the final solution was to hardcode the sigmoid as the activation function, as this is the one
    we wanted to use throughout the code. 
    """
    a = inputs
    activations = [inputs]
    zs = []
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = a @ W + b
        zs.append(z)
        # a = activation_func(z)
        #a = ACTIVATIONS[activation_func](z)
        a = sigmoid(z)
        activations.append(a)
        
    return zs, activations



def predict(X, layers, hidden_activation="sigmoid", head="mse"):
    return feed_forward_batch(X, layers, hidden_activation)[1][-1]

# hidden activations: (f, f')
ACTIVATIONS = {"sigmoid": (sigmoid, lambda z: sigmoid(z) * (1 - sigmoid(z))),
               "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
               "relu": (ReLU, lambda z: (z > 0).astype(float)),
               "identity": (lambda z: z, lambda z: np.ones_like(z))}

def cost(X, Y, layers, hidden_activation="sigmoid", head="mse"):
    out = predict(X, layers, hidden_activation, head)
    if head == "mse":
        return 0.5 * np.sum((out - Y) ** 2) / len(X)
    p = np.clip(out, 1e-12, 1 - 1e-12)
    if head == "binary":
        return -np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p))
    return -np.mean(np.sum(Y * np.log(p), axis=1))


def backpropagation(X, Y, layers, hidden_activation="sigmoid", head="mse"):
    # The four equations, for all samples at once. Returns [(dW, db), ...] with the shapes of [(W, b), ...].
    act_function, fp = ACTIVATIONS[hidden_activation]
    print(type(act_function), type(fp))

    activation_func = ACTIVATIONS[hidden_activation][0]
    activation_funcs = [activation_func] * len(layers)
    z, a = feed_forward_batch(X, layers, activation_funcs)
    
    delta = (a[-1] - Y) / len(X)                         # output layer: prediction minus target
    grads = [None] * len(layers)
    for l in range(len(layers) - 1, -1, -1):
        grads[l] = (a[l].T @ delta, np.sum(delta, axis=0))   # dC/dW = A^T Delta,  dC/db = sum of rows
        if l > 0:
            W = layers[l][0]
            delta = (delta @ W.T) * fp(z[l - 1])         # one layer back
    return grads

def train(X, Y, layers, hidden_activation="sigmoid", head="mse", gamma=0.1, epochs=100,
          batch_size=None, rng=None, X_test=None, y_test=None):
    # Plain gradient descent on minibatches (batch_size=None: all data in every step).
    rng = np.random.default_rng(2026) if rng is None else rng
    n = len(X)
    M = n if batch_size is None else batch_size
    history = {"cost": [], "test_accuracy": []}
    for epoch in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, M):
            idx = order[s:s + M]
            grads = backpropagation(X[idx], Y[idx], layers, hidden_activation, head)
            layers = [(W - gamma * dW, b - gamma * db) for (W, b), (dW, db) in zip(layers, grads)]
        history["cost"].append(cost(X, Y, layers, hidden_activation, head))
        if X_test is not None:
            pred = np.argmax(predict(X_test, layers, hidden_activation, head), axis=1)
            history["test_accuracy"].append(np.mean(pred == y_test))
    return layers, history


def one_hot(y, n_classes):
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y

"""
BLUE = "#2a8fb3"
GREEN = "#2ab37a"
RED = "#b3352a"
GREY = "#77797a"

z = np.linspace(-5, 5, 400)
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
for name, col, ls in (("sigmoid", BLUE, "-"), ("tanh", GREEN, "--"), ("relu", RED, "-")):
    f, fp = ACTIVATIONS[name]
    ax[0].plot(z, f(z), color=col, lw=2.2, ls=ls, label=name)
    ax[1].plot(z, fp(z), color=col, lw=2.2, ls=ls, label=name)
ax[0].set_ylim(-1.2, 3)
ax[0].set_title("activation $f(z)$")
ax[1].set_title("derivative $f'(z)$")
ax[1].axhline(0.25, color=GREY, lw=0.8, ls=":")
ax[1].text(-5, 0.27, "1/4", color=GREY, fontsize=11)
for a_ in ax:
    a_.set_xlabel("$z$")
ax[0].legend(frameon=False)
fig.tight_layout()
plt.show(fig, "activations.png")
"""

from sklearn.model_selection import train_test_split

d = 8                         # your digit
DATA = np.load("mnist.npz")            # see Project 2 for where to get it
x_train = DATA["x_train"].reshape(len(DATA["x_train"]), -1) / 255.0
x_test = DATA["x_test"].reshape(len(DATA["x_test"]), -1) / 255.0
y_train = (DATA["y_train"]).astype(int)
y_test = (DATA["y_test"]).astype(int)

print(np.shape(x_train))
print(np.shape(x_test))

print(np.shape(DATA["x_train"]))


print(f"train {len(y_train)}, test {len(y_test)} images")
print(f"fraction of eights: train {y_train.mean():.3f}, test {y_test.mean():.3f}")
print(f"accuracy of 'never an eight' on the test set: {1 - y_test.mean():.4f}")

# WE plot a selection of the data points for fun!!
number_of_handwritten_figures_to_be_displayed = 12
fig, ax = plt.subplots(1, number_of_handwritten_figures_to_be_displayed , figsize=(10, 1.6))
for a, i in zip(ax, np.where(y_test == 1)[0][:number_of_handwritten_figures_to_be_displayed ]):
    a.imshow(x_test[i].reshape(28, 28), cmap="gray_r"); a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"{number_of_handwritten_figures_to_be_displayed} test images that are the digit {d}", fontsize=9)
plt.show()

t0 = time.time()
layers = create_layers_batch(784, [100, 10])
# layers, hist = train(x_train, one_hot(y_train, 10), layers, "relu", head="mse", gamma=0.1, epochs=20,
#                      batch_size = 100, X_test = x_test, y_test = y_test)
layers, hist = train(x_train, one_hot(y_train, 10), layers, hidden_activation="sigmoid", head="mse", gamma=0.1, epochs=100,
          batch_size = 100, rng=None, X_test = x_test, y_test = y_test)
print(f"test accuracy after 20 epochs: {hist['test_accuracy'][-1]:.2%}  ({time.time() - t0:.0f} s)")

P = predict(x_test, layers, "relu", "softmax")
pred, pmax = P.argmax(axis=1), P.max(axis=1)
# i_A = int(np.argmax(np.where(pred == y_test, pmax, -1)))     # the most confident correct prediction
# i_B = int(np.argmin(pmax))                                   # the least confident prediction

k = np.random.choice(len(x_test), size=10, replace=False)

number_of_images = 10

RED = "#DA2488"

fig = plt.figure(figsize=(6.2, 1.7 * number_of_images))
gs = fig.add_gridspec(number_of_images, 2, width_ratios=[1, 2.6], wspace=0.45, hspace=0.5)
for row, i in enumerate((k)):
    axi = fig.add_subplot(gs[row, 0])
    axi.imshow(x_test[i].reshape(28, 28), cmap="gray_r")
    axi.set_title(f"label {y_test[i]}")
    axi.axis("off")
    axi.set_xticks([])
    axi.set_yticks([])
    axb = fig.add_subplot(gs[row, 1])
    axb.bar(np.arange(10), P[i], color=RED, width=0.7)
    axb.set_xticks(np.arange(10))
    axb.set_ylim(0, 1.05)
    axb.set_ylabel("p")
    axb.set_title(f"image {i}")
axb.set_xlabel("class $k$")
plt.show()
plt.savefig("mnist_question.png")

alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I","K"]

for row, i in enumerate((k)):
    delta = P[row] - one_hot(y_test[row:row + 1], 10)[0]
    print(f"image {i} (label {y_test[i]}): delta^L = a - y = {np.round(delta, 3)}")

# y_B = one_hot(y_test[i_B:i_B + 1], 2)[0]
# delta_B = P[i_B] - y_B
# print(f"\nimage B: label {y_test[i_B]},  one-hot y = {y_B.astype(int)}")
# for k in np.argsort(-np.abs(delta_B)):                       # largest |delta_k| first
#     print(f"  delta_{k} = a_{k} - y_{k} = {P[i_B, k]:.2f} - {y_B[k]:.0f} = {delta_B[k]:+.2f}")

from sklearn.linear_model import LogisticRegression
# Initialize the model (multinomial logistic regression with the saga solver)
model = LogisticRegression(solver='saga', max_iter=1000, random_state=42)
# Train the model
model.fit(x_train, y_train)
from sklearn.metrics import accuracy_score
# Make predictions on the test set
y_pred = model.predict(x_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")