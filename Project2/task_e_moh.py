import os
import time
import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)     # double precision, so that gradient checks agree to ~1e-15
import jax.numpy as jnp
from sklearn.metrics import accuracy_score

rng = np.random.default_rng(2026)

np.random.default_rng(2026)


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
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=False))
    return e_z / jnp.sum(e_z, axis=1, keepdims=True)

def create_layers_batch(network_input_size, layer_output_sizes):
    layers = []
    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = rng.normal(0, 1/np.sqrt(i_size), (i_size, layer_output_size))
        b = np.zeros(layer_output_size) + 0.01
        layers.append((W, b))

        i_size = layer_output_size
    return layers

def feed_forward_batch(inputs, layers, hidden_activation="sigmoid"):
    """
    LLM tag: Gave ChatGPT some of my functions, as there were some issues with the dictionary used for the activation functions.
    The LLM helped debug some lines. However, the final solution was to hardcode the sigmoid as the activation function, as this is the one
    we wanted to use throughout the code. More debugging was used, solution was to fix string vs activation funciton.
    """
    activation_func = ACTIVATIONS[hidden_activation][0]

    a = inputs
    activations = [inputs]
    zs = []
    for W, b in layers:
        z = a @ W + b
        zs.append(z)
        a=activation_func(z)
        activations.append(a)
        
    return zs, activations


def predict(X, layers, hidden_activation="sigmoid", head="mse"):
    return feed_forward_batch(X, layers, hidden_activation)[1][-1]

# hidden activations: (f, f')
ACTIVATIONS = {"sigmoid": (sigmoid, lambda z: sigmoid(z) * (1 - sigmoid(z))),
               "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
               "relu": (ReLU, lambda z: (z > 0).astype(float)),
               "lrelu": (leaky_ReLU, leaky_ReLU_derivative),
               "identity": (lambda z: z, lambda z: np.ones_like(z))}

def cost(X, Y, layers, hidden_activation="sigmoid", head="mse", norm="l2", lam=0.0):
    out = predict(X, layers, hidden_activation, head)

    if norm == "l1":
        norm_value = lam * sum(np.sum(np.abs(W)) for W, _ in layers)
    elif norm == "l2":
        norm_value = lam * sum(np.sum(W**2) for W, _ in layers)
    else:
        raise ValueError("norm must be 'l1' or 'l2'")

    if head == "mse":
        data_loss = 0.5 * np.sum((out - Y) ** 2) / len(X)
    else:
        p = np.clip(out, 1e-12, 1 - 1e-12)
        if head == "binary":
            data_loss = -np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p))
        else:
            data_loss = -np.mean(np.sum(Y * np.log(p), axis=1))
    return data_loss + norm_value


def backpropagation(X, Y, layers, hidden_activation="sigmoid", head="mse", norm="l2", lam=0.0):
    # The four equations, for all samples at once. Returns [(dW, db), ...] with the shapes of [(W, b), ...].
    act_function, fp = ACTIVATIONS[hidden_activation]

    z, a = feed_forward_batch(X, layers, hidden_activation)
    
    delta = (a[-1] - Y) / len(X)                         # output layer: prediction minus target
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

def train(X, Y, layers, hidden_activation="sigmoid", head="mse", norm="l1", lam=0.0, gamma=0.1, epochs=100,
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
            grads = backpropagation(X[idx], Y[idx], layers, hidden_activation, head, norm, lam)
            layers = [(W - gamma * dW, b - gamma * db) for (W, b), (dW, db) in zip(layers, grads)]
        history["cost"].append(cost(X, Y, layers, hidden_activation, head, norm, lam))
        if X_test is not None:
            pred = np.argmax(predict(X_test, layers, hidden_activation, head), axis=1)
            history["test_accuracy"].append(np.mean(pred == y_test))
    return layers, history


def accuracy(predictions, targets):
    """Fraction of rows whose largest predicted probability sits in the correct class."""
    return accuracy_score(np.argmax(np.asarray(predictions), axis=1), np.argmax(np.asarray(targets)))


def one_hot(y, n_classes):
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y

from sklearn.model_selection import train_test_split

d = 8
DATA = np.load("Project2/mnist.npz")
x_train = DATA["x_train"].reshape(len(DATA["x_train"]), -1) / 255.0
x_test = DATA["x_test"].reshape(len(DATA["x_test"]), -1) / 255.0
y_train = (DATA["y_train"]).astype(int)
y_test = (DATA["y_test"]).astype(int)


print(f"train {len(y_train)}, test {len(y_test)} images")
print(f"fraction of eights: train {y_train.mean():.3f}, test {y_test.mean():.3f}")
print(f"accuracy of 'never an eight' on the test set: {1 - y_test.mean():.4f}")

# WE plot a selection of the data points for fun!!
number_of_handwritten_figures_to_be_displayed = 6
fig, ax = plt.subplots(1, number_of_handwritten_figures_to_be_displayed , figsize=(10, 1.6))
for a, i in zip(ax, np.where(y_test == 1)[0][:number_of_handwritten_figures_to_be_displayed ]):
    a.imshow(x_test[i].reshape(28, 28), cmap="gray_r"); a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"{number_of_handwritten_figures_to_be_displayed} test images that are the digit {d}", fontsize=9)
plt.show()


plt.rcParams.update({'font.size': 13})
# Må fikse plot.
plt.figure()
for act_func in ["sigmoid", "tanh", "relu", "lrelu"]:
    t0 = time.time()
    layers = create_layers_batch(784, [100, 10])
    layers, hist = train(x_train, one_hot(y_train, 10), layers, hidden_activation=act_func, head="mse", gamma=0.1, epochs=50,
            batch_size = 100, rng=None, X_test = x_test, y_test = y_test, norm="l2", lam=0.0)
    print(f"test accuracy after 50 epochs for {act_func}: {hist['test_accuracy'][-1]:.2%}  ({time.time() - t0:.0f} s)")
    plt.plot(np.arange(1, len(hist["test_accuracy"])+1), hist["test_accuracy"], label=f"{act_func}")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.grid()
plt.legend()
plt.savefig("Project2/activation_funcs_accuracy.pdf", dpi=300)
plt.show()

lambda_values = np.logspace(-3, 0, 10)
# Må fikse plot.
plt.figure()

for lam in lambda_values:
    t0 = time.time()
    layers = create_layers_batch(784, [100, 10])
    layers, hist = train(x_train, one_hot(y_train, 10), layers, hidden_activation="lrelu", head="mse", gamma=0.1, epochs=50,
            batch_size = 100, rng=None, X_test = x_test, y_test = y_test, norm="l2", lam=0.0)
    print(f"test accuracy after 50 epochs for LReLU and lam={lam}: {hist['test_accuracy'][-1]:.2%}  ({time.time() - t0:.0f} s)")
    plt.plot(np.arange(1, len(hist["test_accuracy"])+1), hist["test_accuracy"], label=f"lam={lam}")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.grid()
plt.legend()
plt.savefig("Project2/l2_accuracy.pdf", dpi=300)
plt.show()


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
fig.savefig("Project2/mnist_question.pdf", dpi=300)

for row, i in enumerate((k)):
    delta = P[row] - one_hot(y_test[row:row + 1], 10)[0]
    print(f"image {i} (label {y_test[i]}): delta^L = a - y = {np.round(delta, 3)}")

exit()

from sklearn.linear_model import LogisticRegression
# Initialize the model (multinomial logistic regression with the saga solver)
model = LogisticRegression(solver='saga', max_iter=1000, random_state=42)
# Train the model
model.fit(x_train, y_train)
# Make predictions on the test set
y_pred = model.predict(x_test)
# Calculate accuracy
accuracy1 = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy1:.4f}")