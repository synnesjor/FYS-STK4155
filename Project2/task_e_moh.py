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
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True)) # LLM-tag (Claude Code) make this True
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

def feed_forward_batch(inputs, layers, hidden_activation="sigmoid", head="binary"):
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


def predict(X, layers, hidden_activation="sigmoid", head="binary"):
    return feed_forward_batch(X, layers, hidden_activation, head)[1][-1]

# hidden activations: (f, f')
ACTIVATIONS = {"sigmoid": (sigmoid, lambda z: sigmoid(z) * (1 - sigmoid(z))),
               "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
               "relu": (ReLU, lambda z: (z > 0).astype(float)),
               "lrelu": (leaky_ReLU, leaky_ReLU_derivative),
               "identity": (lambda z: z, lambda z: np.ones_like(z))}

def cost(X, Y, layers, hidden_activation="sigmoid", head="binary", norm="l2", lam=0.0):
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

    z, a = feed_forward_batch(X, layers, hidden_activation)
    if head == "mse":
        delta = 2*(a[-1] - Y) / len(X)
    elif head in ["binary", "cross"]:
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

def train(X, Y, layers, hidden_activation="sigmoid", head="cross", norm="l1", lam=0.0, gamma=0.1, epochs=100, method="adam", rng=None, batch_size=None, X_test=None, y_test=None, tol=1e-6):
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
            layers = new_layers
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

from sklearn.model_selection import StratifiedKFold


def cross_validate_neural_network(X, y, hidden_sizes=[100], epochs=50, n_splits=5, norm="l2", lam=0.0, gamma=0.1):
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
            method="plain",
            batch_size=100,
            epochs=epochs,
            rng=2026 + fold,
            X_test=X_val_fold,
            y_test=y_val_fold
        )

        score = history["test_accuracy"][-1]
        fold_scores.append(score)

        print(f"Fold {fold + 1}: {score:.4f}")

    return np.array(fold_scores)

plt.rcParams.update({'font.size': 13})


d = 8
DATA = np.load("Project2/mnist.npz")
x_train = DATA["x_train"].reshape(len(DATA["x_train"]), -1) / 255.0
x_test = DATA["x_test"].reshape(len(DATA["x_test"]), -1) / 255.0
y_train = (DATA["y_train"]).astype(int)
y_test = (DATA["y_test"]).astype(int)

train_size = 10000  # choose any smaller size
indices = np.random.default_rng(2026).choice(
    len(DATA["x_train"]),
    size=train_size,
    replace=False
)

small_x_train = DATA["x_train"][indices].reshape(train_size, -1) / 255.0
small_y_train = DATA["y_train"][indices].astype(int)

# First tune architecture:

select_architecture_gamma = False

while select_architecture_gamma:
    architectures = [[50], [100], [100, 10], [100, 50]]
    gammas = np.linspace(0.01, 0.5, 10)
    best_parameters = None
    best_accuracy = 0
    for architecture in architectures:
        for gamma in gammas:
            scores = cross_validate_neural_network(
                small_x_train, small_y_train, hidden_sizes=architecture,
                epochs=50, n_splits=3, lam=0.0, gamma=gamma
            )
            print(architecture, gamma, f"mean={scores.mean():.4f}", f"SD={scores.std():.4f}")
            if scores.mean() > best_accuracy:
                best_parameters = (architecture, gamma)
                best_accuracy = scores.mean()
    print(f"Best Architecture: {best_parameters[0]}; Best Learning Rate: {best_parameters[1]}")
    select_architecture_gamma = False

best_architecture = [100, 50]

select_learning_rate = False
# Selected using smaller training set (10000):
while select_learning_rate:
    gammas = np.linspace(0.01, 0.6, 31)
    best_gamma = None
    best_accuracy = 0
    for gamma in gammas:
        t0 = time.time()
        scores = cross_validate_neural_network(
            small_x_train, small_y_train, hidden_sizes=best_architecture,
            epochs=50, n_splits=3, lam=0.0, gamma=gamma
        )
        print(f"gamma={gamma}", f"mean={scores.mean():.4f}", f"SD={scores.std():.4f}", f"({round(time.time()-t0)}")
        if best_accuracy < scores.mean():
            best_gamma = gamma
            best_accuracy = scores.mean()
    
    print(f"Best learning rate: gamma={best_gamma}")
    select_learning_rate = False
    
gamma_adam = 0.01584893192461114
gamma_rms = 0.002511886431509582
gamma_plain = 0.22633333333333333

select_activation_function = False

# Selected using full training data, discarded.
while select_activation_function:
    # Må fikse plot.
    plt.figure()
    for act_func in ["sigmoid", "tanh", "relu", "lrelu"]:
        for method, gamma in [("plain", gamma_plain), ("rmsprop", gamma_rms), ("adam", gamma_adam)]:
            t0 = time.time()
            layers = create_layers_batch(784, best_architecture + [10])
            layers, hist = train(x_train, one_hot(y_train, 10), layers, hidden_activation=act_func, head="cross", method=method, gamma=gamma, epochs=50,
                    batch_size = 100, rng=None, X_test = x_test, y_test = y_test, norm="l2", lam=0.0)
            print(f"test accuracy after 50 epochs for {act_func} and {method}: {hist['test_accuracy'][-1]:.2%}  ({time.time() - t0:.0f} s)")
            plt.plot(np.arange(1, len(hist["test_accuracy"])+1), hist["test_accuracy"], label=f"{act_func}, {method}")

    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.grid()
    plt.legend()
    plt.savefig("Project2/activation_funcs_accuracy.pdf", dpi=300)
    plt.show()
    select_activation_function = False

select_accuracy_analysis = False
# Selected using full training set
while select_accuracy_analysis:
    save_accuracy = {"sigmoid": [], "tanh": [], "relu": [], "lrelu": []}
    for idx in range(5):
        for act_func in ["sigmoid", "tanh", "relu", "lrelu"]:
            t0 = time.time()
            layers = create_layers_batch(784, best_architecture + [10])
            layers, hist = train(x_train, one_hot(y_train, 10), layers, hidden_activation=act_func, head="cross", method="plain", gamma=gamma_plain, epochs=50,
                    batch_size = 100, rng=2026+idx, X_test = x_test, y_test = y_test, norm="l2", lam=0.0)
            print(f"test accuracy after 50 epochs for plain and {act_func}: {hist['test_accuracy'][-1]:.2%}  ({time.time() - t0:.0f} s)")
            save_accuracy[act_func].append(hist['test_accuracy'][-1])

    for act_func in ["sigmoid", "tanh", "relu", "lrelu"]:
        print(f"{act_func} mean: {np.mean(save_accuracy[act_func])}, SD: {np.std(save_accuracy[act_func])}")
    activation_names = ["sigmoid", "tanh", "relu", "lrelu"]
    means = np.array([np.mean(save_accuracy[name]) for name in activation_names])
    stds = np.array([np.std(save_accuracy[name]) for name in activation_names])

    with open("Project2/activation_accuracy.txt", "w", encoding="utf-8") as f:
        for mean, sd in zip(means, stds):
            f.write(f"{mean:.16g} {sd:.16g}\n")
    
    select_accuracy_analysis = False

display_names = ["sigmoid", "tanh", "ReLU", "LReLU"]

data_accuracy = np.loadtxt("Project2/activation_accuracy.txt", delimiter=" ")
data_accuracy = data_accuracy.T
means = data_accuracy[0]
stds = data_accuracy[1]

fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(means))

ax.errorbar(
    x, means, yerr=stds,
    fmt="o", markersize=8, linewidth=1.8,
    color="#0072B2", ecolor="#0072B2",
    capsize=6
)

ax.set_xticks(x, labels=display_names, rotation=25, ha="right")
ax.set_ylabel("Accuracy")
ax.grid(True, alpha=0.3)

# Limits the y-axis
padding = max(3 * stds.max(), 0.005)
ax.set_ylim(max(0, means.min() - padding), min(1, means.max() + padding))

for mean in means:
    ax.axhline(mean, ls="--", alpha=0.4, color="black")

fig.tight_layout()
fig.savefig("Project2/activation_accuracy_seeds.pdf",
            dpi=300, bbox_inches="tight")
plt.show()

select_hyperparameter = False
# Selected using smaller training set, 10000
while select_hyperparameter: # do for l1 and l2
    lambda_values = np.linspace(0, 1e-2, 50)  
    save_accuracy = []
    norm = "l1"
    for lam in lambda_values:
        t0 = time.time()
        layers = create_layers_batch(784, best_architecture + [10])
        layers, hist = train(small_x_train, one_hot(small_y_train, 10), layers, hidden_activation="relu", head="cross", method="plain", gamma=gamma_plain, epochs=50,
                batch_size = 100, rng=None, X_test = x_test, y_test = y_test, norm=norm, lam=lam)
        print(f"test accuracy after 50 epochs for ReLU and lam={lam}: {hist['test_accuracy'][-1]:.2%}  ({time.time() - t0:.0f} s)")
        save_accuracy.append(hist["test_accuracy"][-1])


    with open(f"Project2/{norm}_accuracy.txt", "w", encoding="utf-8") as f:
        for lam, accuracy_value in zip(lambda_values, save_accuracy):
            f.write(f"{lam} {accuracy_value:.16g}\n")

    select_hyperparameter = False

data_hyperparameter_l2 = np.loadtxt("Project2/l2_accuracy.txt", delimiter=" ")
data_hyperparameter_l2 = data_hyperparameter_l2.T
lambda_values_l2 = data_hyperparameter_l2[0]
accuracy_values_l2 = data_hyperparameter_l2[1]
plt.figure()
plt.plot(lambda_values_l2, accuracy_values_l2, label="l2")
plt.ylabel("Accuracy")
plt.xlabel("Lambda")

data_hyperparameter_l1 = np.loadtxt("Project2/l1_accuracy.txt", delimiter=" ")
data_hyperparameter_l1 = data_hyperparameter_l1.T
lambda_values_l1 = data_hyperparameter_l1[0]
accuracy_values_l1 = data_hyperparameter_l1[1]
plt.plot(lambda_values_l1, accuracy_values_l1, label="l1")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("Project2/norm_accuracy.pdf", dpi=300)

plt.show()

select_baseline_model = True

while select_baseline_model:
    t0 = time.time()
    lam_adam = 1e-4
    layers = create_layers_batch(784, [10])
    layers, hist = train(x_train, one_hot(y_train, 10), layers, hidden_activation="relu", head="cross", method="plain", gamma=gamma_plain, epochs=50,
                    batch_size = 100, rng=None, X_test = x_test, y_test = y_test, norm="l2", lam=0.0)

    print(f"test accuracy after 50 epochs for ReLU and lam={0.0}: {hist['test_accuracy'][-1]:.2%}  ({time.time() - t0:.0f} s)")

    select_baseline_model = False


select_best_model = False
# Use the best model, plain, relu, gamma=gamma_plain, no norm
while select_best_model:
    t0 = time.time()
    lam_adam = 1e-4
    layers = create_layers_batch(784, best_architecture + [10])
    layers, hist = train(x_train, one_hot(y_train, 10), layers, hidden_activation="relu", head="cross", method="plain", gamma=gamma_plain, epochs=50,
                    batch_size = 100, rng=None, X_test = x_test, y_test = y_test, norm="l2", lam=0.0)

    print(f"test accuracy after 50 epochs for ReLU and lam={0.0}: {hist['test_accuracy'][-1]:.2%}  ({time.time() - t0:.0f} s)")

    P = predict(x_test, layers, "relu", "cross")
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

    for row, i in enumerate(k):
        delta = P[i] - one_hot(y_test[i:i + 1], 10)[0]
        print(f"image {i} (label {y_test[i]}): " f"delta^L = a - y = {np.round(delta, 3)}")

    # LLM-tag - easy way to save the data from the model:
    # Save all weights and biases.
    np.savez(
        "Project2/best_model_layers.npz",
        P=P,
        **{
            f"layer_{i}_{name}": value
            for i, (W, b) in enumerate(layers)
            for name, value in (("W", W), ("b", b))
        }
    )

    # Save history as: epoch, cost, test accuracy.
    np.savetxt(
        "Project2/best_model_history.txt",
        np.column_stack((
            np.arange(1, len(hist["cost"]) + 1),
            hist["cost"],
            hist["test_accuracy"]
        )),
        header="epoch cost test_accuracy",
        comments=""
    )
    
    select_best_model = False


# LLM-tag: reading saved data
# Read history
history_data = np.loadtxt("Project2/best_model_history.txt",skiprows=1).T
history = {"epoch": history_data[0],"cost": history_data[1],"test_accuracy": history_data[2]}

# Read layers.
saved_model = np.load("Project2/best_model_layers.npz")
P = saved_model["P"]

layer_indices = sorted({int(name.split("_")[1]) for name in saved_model.files if name.startswith("layer_")})

layers = [(saved_model[f"layer_{i}_W"], saved_model[f"layer_{i}_b"]) for i in layer_indices]

from sklearn.metrics import (confusion_matrix as sk_confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, f1_score,
                             classification_report as sk_classification_report)

select_confusion_matrix = True

while select_confusion_matrix:

    p_test = np.argmax(P, axis=1)
    y_true = np.asarray(y_test).ravel()

    cm = sk_confusion_matrix(y_true, p_test)
    accuracy = accuracy_score(y_true, p_test)

    print(cm)
    print(f"Accuracy: {accuracy:.4f}")
    print("Precision:", precision_score(y_true, p_test, average="macro", zero_division=0))
    print("Recall:", recall_score(y_true, p_test, average="macro", zero_division=0))
    print("F1:", f1_score(y_true, p_test, average="macro", zero_division=0))
    print(sk_classification_report(y_true, p_test, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(y_true, p_test, ax=ax, colorbar=False)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

    select_confusion_matrix = False


select_sklearn_model = False
# Running code gives 92.57% accuracy
while select_sklearn_model: 
    from sklearn.linear_model import LogisticRegression
    # Initialize the model (multinomial logistic regression with the saga solver)
    model = LogisticRegression(solver='saga', max_iter=1000, random_state=42)
    # Train the model
    model.fit(x_train, y_train)
    # Make predictions on the test set
    y_pred = model.predict(x_test)
    # Calculate accuracy
    accuracy_value = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy_value:.4f}") 
    
    select_sklearn_model = False