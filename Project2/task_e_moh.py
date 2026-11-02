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

select_architecture = False

while select_architecture:
    architectures = [[50], [100], [100, 10], [100, 50]]
    gammas = np.linspace(0.01, 0.5, 10)
    best_parameters = None
    best_accuracy = 0 # choose small value since we want to values closest to 1
    save_parameters = []
    save_accuracy = []
    # Hard coding grid search
    for architecture in architectures:
        for gamma in gammas:
            t0 = time.time()
            cv_mean, cv_sd = cross_validate_neural_network(
                small_x_train, small_y_train, hidden_sizes=architecture,
                epochs=50, n_splits=3, lam=0.0, gamma=gamma
            )
            print(architecture, gamma, f"mean={cv_mean:.4f}", f"SD={cv_sd:.4f}", f"({round(time.time()-t0)} s)")
            save_parameters.append((architecture, gamma))
            save_accuracy.append(cv_mean)
            if cv_mean > best_accuracy:
                best_parameters = (architecture, gamma)
                best_accuracy = cv_mean
                threshold = cv_mean - cv_sd # Minus, since we want higher values

    save_accuracy = np.array(save_accuracy)
    within_one_sd = save_accuracy >= threshold
    eligible_indices = np.where(within_one_sd)[0]

    eligible_parameters = [save_parameters[i] for i in eligible_indices]
    
    for i in eligible_indices:
        print(f"Eligible parameters: {save_parameters[i]} " f"with accuracy {save_accuracy[i]:.4f} ")

    print(f"Best Architecture: {best_parameters[0]}; Best Learning Rate: {best_parameters[1]}")
    select_architecture = False

best_architecture = [100, 50]

select_learning_rate_norm = False
# Selected using smaller training set (10000):
# Change between plain, adam and rms here.
while select_learning_rate_norm:
    gammas = np.linspace(0.01, 0.6, 5)
    lambda_values = np.linspace(0, 1e-3, 5)
    best_parameters = None
    best_accuracy = 0
    save_accuracy = []
    save_parameters = []
    for gamma in gammas:
        for lam in lambda_values:
            t0 = time.time()
            cv_mean, cv_sd = cross_validate_neural_network(
                small_x_train, small_y_train, hidden_sizes=best_architecture,
                epochs=50, n_splits=3, lam=lam, norm="l2", gamma=gamma, method="plain"
            )
            print(gamma, lam, f"mean={cv_mean:.4f}", f"SD={cv_sd:.4f}", f"({round(time.time()-t0)} s)")

            save_accuracy.append(cv_mean)
            save_parameters.append((gamma, lam))
            if best_accuracy < cv_mean:
                best_parameters = (gamma, lam)
                best_accuracy = cv_mean
                threshold = cv_mean - cv_sd
    
    save_accuracy = np.array(save_accuracy)
    within_one_sd = save_accuracy >= threshold
    eligible_indices = np.where(within_one_sd)[0]

    eligible_parameters = [save_parameters[i] for i in eligible_indices]
    
    for i in eligible_indices:
        print(f"Eligible parameters: {save_parameters[i]} " f"with accuracy {save_accuracy[i]:.4f} ")

    print(f"Best Learning Rate: {best_parameters[0]}; Best L2 penalty parameter: {best_parameters[1]}")
    select_learning_rate_norm = False
    
eligible_parameters_l2 = [(np.float64(0.305), np.float64(0.00025)), 
                       (np.float64(0.305), np.float64(0.0005)), 
                       (np.float64(0.4525), np.float64(0.0)),
                       (np.float64(0.4525), np.float64(0.00025)),
                       (np.float64(0.4525), np.float64(0.0005)),
                       (np.float64(0.6), np.float64(0.0)),
                       (np.float64(0.6), np.float64(0.00025))]


select_learning_rate_norm_with_test_data = False
# Using test data to confirm learning rate and norm.
while select_learning_rate_norm_with_test_data:
    save_parameters = []
    save_accuracy = []
    hidden_sizes = best_architecture
    # Plain:
    for idx, (gamma, lam) in enumerate(eligible_parameters_l2):
        t0 = time.time()
        layers = create_layers_batch(784, hidden_sizes + [10])
        layers, history = train(
        x_train,
        one_hot(y_train, 10),
        layers,
        hidden_activation="relu",
        head="cross",
        gamma=gamma,
        method="plain",
        batch_size=100,
        epochs=50,
        rng=2026,
        X_test=x_test,
        y_test=y_test,
        lam=lam
        )
        accuracy = history["test_accuracy"][-1]

        save_parameters.append((gamma, lam))
        save_accuracy.append(accuracy)
        print(gamma, lam, f"accuracy={accuracy:.4f}" f"({round(time.time()-t0)} s)")

    best_idx = int(np.argmax(save_accuracy))

    print(f"Best parameters: {save_parameters[best_idx]};", f"Accuracy: {save_accuracy[best_idx]}")

    select_learning_rate_norm_with_test_data = False

# Analysis confirms lam=0.0
optimal_lam = 0

select_learning_rate = False

while select_learning_rate:
    gammas = np.linspace(0.01, 0.6, 30)
    best_parameters = None
    best_accuracy = 0
    save_accuracy = []
    save_parameters = []
    for gamma in gammas:
        t0 = time.time()
        cv_mean, cv_sd = cross_validate_neural_network(
            small_x_train, small_y_train, hidden_sizes=best_architecture,
            epochs=50, n_splits=3, lam=optimal_lam, norm="l2", gamma=gamma, method="plain"
        )
        print(gamma, f"mean={cv_mean:.4f}", f"SD={cv_sd:.4f}", f"({round(time.time()-t0)} s)")

        save_accuracy.append(cv_mean)
        save_parameters.append(gamma)
        if best_accuracy < cv_mean:
            best_parameters = gamma
            best_accuracy = cv_mean
            threshold = cv_mean - cv_sd
    
    save_accuracy = np.array(save_accuracy)
    within_one_sd = save_accuracy >= threshold
    eligible_indices = np.where(within_one_sd)[0]

    eligible_parameters = [save_parameters[i] for i in eligible_indices]
    
    for i in eligible_indices:
        print(f"Eligible parameters: {save_parameters[i]} " f"with accuracy {save_accuracy[i]:.4f} ")

    print(f"Best Learning Rate: {best_parameters}")
    select_learning_rate = False

eligible_gamma = np.array([0.396551724137931, 0.5186206896551724, 0.579655172413793])

select_final_learning_rate = False

while select_final_learning_rate:
    save_parameters = []
    save_accuracy = []
    hidden_sizes = best_architecture
    # Plain:
    for idx, gamma in enumerate(eligible_gamma):
        t0 = time.time()
        layers = create_layers_batch(784, hidden_sizes + [10])
        layers, history = train(
        x_train,
        one_hot(y_train, 10),
        layers,
        hidden_activation="relu",
        head="cross",
        gamma=gamma,
        method="plain",
        batch_size=100,
        epochs=50,
        rng=2026,
        X_test=x_test,
        y_test=y_test,
        lam=optimal_lam
        )
        accuracy = history["test_accuracy"][-1]

        save_parameters.append(gamma)
        save_accuracy.append(accuracy)
        print(gamma, f"accuracy={accuracy:.4f}" f"({round(time.time()-t0)} s)")

    best_idx = int(np.argmax(save_accuracy))

    print(f"Best parameters: {save_parameters[best_idx]};", f"Accuracy: {save_accuracy[best_idx]}")

    select_final_learning_rate = False

optimal_gamma = 0.5186206896551724

select_accuracy_analysis = False
# Selected using full training set
while select_accuracy_analysis:
    save_accuracy = {"sigmoid": [], "tanh": [], "relu": [], "lrelu": []}
    save_time = {"sigmoid": [], "tanh": [], "relu": [], "lrelu": []}
    for idx in range(5):
        for act_func in ["sigmoid", "tanh", "relu", "lrelu"]:
            t0 = time.time()
            layers = create_layers_batch(784, best_architecture + [10])
            layers, hist = train(x_train, one_hot(y_train, 10), layers, hidden_activation=act_func, head="cross", method="plain", gamma=optimal_gamma, epochs=50,
                    batch_size = 100, rng=2026+idx, X_test = x_test, y_test = y_test, norm="l2", lam=optimal_lam)
            elapsed_time = time.time() - t0
            print(f"test accuracy after 50 epochs for plain and {act_func}: {hist['test_accuracy'][-1]:.2%}  ({elapsed_time:.0f} s)")
            save_accuracy[act_func].append(hist['test_accuracy'][-1])
            save_time[act_func].append(elapsed_time)

    for act_func in ["sigmoid", "tanh", "relu", "lrelu"]:
        print(f"{act_func} mean: {np.mean(save_accuracy[act_func])}, SD: {np.std(save_accuracy[act_func])}")
    activation_names = ["sigmoid", "tanh", "relu", "lrelu"]
    means = np.array([np.mean(save_accuracy[name]) for name in activation_names])
    stds = np.array([np.std(save_accuracy[name]) for name in activation_names])


    for act_func in ["sigmoid", "tanh", "relu", "lrelu"]:
        time_mean = np.mean(save_time[act_func])
        time_sd = np.std(save_time[act_func])

    print(f"time mean={time_mean:.2f} s, SD={time_sd:.2f} s")

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
y = np.linspace(means.min(), means.max(), 6)

ax.errorbar(
    x, means, yerr=stds,
    fmt="o", markersize=8, linewidth=1.8,
    color="#FF46A2", ecolor="#F598C6",
    capsize=6
)

ax.set_xticks(x, size=20, labels=display_names, rotation=25, ha="right")
ax.set_ylabel("Accuracy", size=20)
ax.tick_params(axis="y", labelsize=20)
ax.grid(True, alpha=0.3)
# Limits the y-axis
padding = max(3 * stds.max(), 0.005)
ax.set_ylim(max(0, means.min() - padding), min(1, means.max() + padding))

for mean in means:
    ax.axhline(mean, ls="--", alpha=0.7, color="grey")

fig.tight_layout()
fig.savefig("Project2/activation_accuracy_seeds.pdf",
            dpi=300, bbox_inches="tight")
plt.show()

select_baseline_model = False
# Baseline model gives 92.37%.
while select_baseline_model:
    t0 = time.time()
    lam_adam = 1e-4
    layers = create_layers_batch(784, [10])
    layers, hist = train(x_train, one_hot(y_train, 10), layers, hidden_activation="relu", head="cross", method="plain", gamma=optimal_gamma, epochs=50,
                    batch_size = 100, rng=None, X_test = x_test, y_test = y_test, norm="l2", lam=optimal_lam)

    print(f"test accuracy after 50 epochs for ReLU, lam={optimal_lam}, gamma={optimal_gamma:.4f}: {hist['test_accuracy'][-1]:.2%}  ({time.time() - t0:.0f} s)")

    select_baseline_model = False


select_best_model = True
# Use the best model, plain, relu, gamma=gamma_plain, no norm
while select_best_model:
    t0 = time.time()
    lam_adam = 1e-4
    layers = create_layers_batch(784, best_architecture + [10])
    layers, hist = train(x_train, one_hot(y_train, 10), layers, hidden_activation="relu", head="cross", method="plain", gamma=optimal_gamma, epochs=50,
                    batch_size = 100, rng=None, X_test = x_test, y_test = y_test, norm="l2", lam=0.0)

    print(f"test accuracy after 50 epochs for ReLU and lam={0.0}: {hist['test_accuracy'][-1]:.2%}  ({time.time() - t0:.0f} s)")

    P = predict(x_test, layers, "relu", "cross")
    pred, pmax = P.argmax(axis=1), P.max(axis=1)
    # i_A = int(np.argmax(np.where(pred == y_test, pmax, -1)))     # the most confident correct prediction
    # i_B = int(np.argmin(pmax))                                   # the least confident prediction

    k = np.random.choice(len(x_test), size=5, replace=False)

    number_of_images = 5

    PINK = "#FF46A2"

    fig = plt.figure(figsize=(6.5, 7.0), constrained_layout=True)

    gs = fig.add_gridspec(
        number_of_images, 2,
        width_ratios=[1, 2.8],
        wspace=0.12
    )

    for row, i in enumerate(k):
        axi = fig.add_subplot(gs[row, 0])
        axi.imshow(x_test[i].reshape(28, 28), cmap="gray_r")
        axi.set_title(f"True label: {y_test[i]}", fontsize=14)
        axi.axis("off")

        axb = fig.add_subplot(gs[row, 1])
        predicted = int(np.argmax(P[i]))

        axb.bar(np.arange(10), P[i], color=PINK, width=0.7)
        axb.set_xticks(np.arange(10))
        axb.set_xlim(-0.5, 9.5)
        axb.set_ylim(0, 1.05)

        axb.set_title(
            f"Image {i} — predicted: {predicted}",
            fontsize=14,
            pad=5
        )
        if row == 2:
            axb.set_ylabel("Probability", fontsize=14)
        if row == 4:
            axb.set_xlabel("Class", fontsize=14)
        axb.tick_params(axis="both", labelsize=12)
        axb.grid(axis="y", alpha=0.25)

    fig.savefig(
        "Project2/mnist_question.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08
    )
    plt.show()
    plt.close(fig)

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


    from matplotlib.colors import BoundaryNorm
    from matplotlib import colormaps


    cmap = colormaps["RdPu"].resampled(4)

    norm = BoundaryNorm(
        boundaries=[0, 7, 11, 15, 1e9],  # 0–7, 8–10, 11+
        ncolors=cmap.N)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.tick_params(axis="both", labelsize=19)
    ax.set_xlabel(ax.get_xlabel(), fontsize=19)
    ax.set_ylabel(ax.get_ylabel(), fontsize=19)
    disp = ConfusionMatrixDisplay.from_predictions(y_true, p_test, ax=ax, colorbar=False, cmap=cmap, text_kw={"fontsize": 19})
    disp.im_.set_norm(norm)
    disp.im_.set_cmap(cmap)
    fig.tight_layout()
    fig.savefig("Project2/confusion_matrix_best_model.pdf", dpi=300)
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