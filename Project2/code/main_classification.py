import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from source.utils_classification import *

SEED = 2026
rng = np.random.default_rng(SEED) #LLM tag: ChatGPT suggested to make this more global, as we were having some issues with reproducibility

plt.rcParams.update({'font.size': 13})


DATA = np.load("data/mnist.npz")
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

if select_architecture:
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

best_architecture = [100, 50]

select_learning_rate_norm = False
# Selected using smaller training set (10000):
# Change between plain, adam and rms here.
if select_learning_rate_norm:
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
    
eligible_parameters_l2 = [(np.float64(0.305), np.float64(0.00025)), 
                       (np.float64(0.305), np.float64(0.0005)), 
                       (np.float64(0.4525), np.float64(0.0)),
                       (np.float64(0.4525), np.float64(0.00025)),
                       (np.float64(0.4525), np.float64(0.0005)),
                       (np.float64(0.6), np.float64(0.0)),
                       (np.float64(0.6), np.float64(0.00025))]


select_learning_rate_norm_with_test_data = False
# Using test data to confirm learning rate and norm.
if select_learning_rate_norm_with_test_data:
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
        rng=None,
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

# Analysis confirms lam=0.0
optimal_lam = 0

select_learning_rate = False

if select_learning_rate:
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

eligible_gamma = np.array([0.396551724137931, 0.5186206896551724, 0.579655172413793])

select_final_learning_rate = False

if select_final_learning_rate:
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
        rng=None,
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


optimal_gamma = 0.5186206896551724

select_accuracy_analysis = False
# Selected using full training set
if select_accuracy_analysis:
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

    with open("data/activation_accuracy.txt", "w", encoding="utf-8") as f:
        for mean, sd in zip(means, stds):
            f.write(f"{mean:.16g} {sd:.16g}\n")
    

display_names = ["sigmoid", "tanh", "ReLU", "LReLU"]

data_accuracy = np.loadtxt("data/activation_accuracy.txt", delimiter=" ")
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
fig.savefig("figures/e)/activation_accuracy_seeds.pdf",
            dpi=300, bbox_inches="tight")
plt.show()

select_baseline_model = False
# Baseline model gives 92.37%.
if select_baseline_model:
    t0 = time.time()
    lam_adam = 1e-4
    layers = create_layers_batch(784, [10])
    layers, hist = train(x_train, one_hot(y_train, 10), layers, hidden_activation="relu", head="cross", method="plain", gamma=optimal_gamma, epochs=50,
                    batch_size = 100, rng=None, X_test = x_test, y_test = y_test, norm="l2", lam=optimal_lam)

    print(f"test accuracy after 50 epochs for ReLU, lam={optimal_lam}, gamma={optimal_gamma:.4f}: {hist['test_accuracy'][-1]:.2%}  ({time.time() - t0:.0f} s)")


select_best_model = True
# Use the best model, plain, relu, gamma=gamma_plain, no norm
if select_best_model:
    t0 = time.time()
    layers = create_layers_batch(784, best_architecture + [10])
    layers, hist = train(x_train, one_hot(y_train, 10), layers, hidden_activation="relu", head="cross", method="plain", gamma=optimal_gamma, epochs=50,
                    batch_size = 100, rng=None, X_test = x_test, y_test = y_test, norm="l2", lam=optimal_lam)

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
        "figures/e)/mnist_question.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08
    )
    plt.show()
    plt.close(fig)

    # LLM-tag - easy way to save the data from the model:
    # Save all weights and biases.
    np.savez(
        "data/best_model_layers.npz",
        P=P,
        **{
            f"layer_{i}_{name}": value
            for i, (W, b) in enumerate(layers)
            for name, value in (("W", W), ("b", b))
        }
    )

    # Save history as: epoch, cost, test accuracy.
    np.savetxt(
        "data/best_model_history.txt",
        np.column_stack((
            np.arange(1, len(hist["cost"]) + 1),
            hist["cost"],
            hist["test_accuracy"]
        )),
        header="epoch cost test_accuracy",
        comments=""
    )
    


# LLM-tag: reading saved data
# Read history
history_data = np.loadtxt("data/best_model_history.txt",skiprows=1).T
history = {"epoch": history_data[0],"cost": history_data[1],"test_accuracy": history_data[2]}

# Read layers.
saved_model = np.load("data/best_model_layers.npz")
P = saved_model["P"]

layer_indices = sorted({int(name.split("_")[1]) for name in saved_model.files if name.startswith("layer_")})

layers = [(saved_model[f"layer_{i}_W"], saved_model[f"layer_{i}_b"]) for i in layer_indices]

from sklearn.metrics import (confusion_matrix as sk_confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score, recall_score, f1_score,
                             classification_report as sk_classification_report)

select_confusion_matrix = True

if select_confusion_matrix:

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
    fig.savefig("figures/e)/confusion_matrix_best_model.pdf", dpi=300)
    plt.show()
    plt.close(fig)



select_sklearn_model = False
# Running code gives 92.57% accuracy
if select_sklearn_model: 
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