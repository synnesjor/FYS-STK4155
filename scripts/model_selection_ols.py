import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.utils import resample
from src.utils import *

def main(x, y):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    polynomial_degree = 16
    degrees = np.arange(polynomial_degree)
    folds_list = [5, 10, 25]
    number_of_repeats = 100
    colors = ["#aa333c", "#99d2fb", "#9aab64"]
    decomposition = {}
    optimal_parameters = {}

    for folds in folds_list:
        test_error = []
        bias = []
        variance = []

        for degree in degrees:
            predictions = np.zeros((len(y), number_of_repeats))
            for repeat in range(number_of_repeats):
                kfold = KFold(
                    n_splits=folds,
                    shuffle=True,
                    random_state=2026 + repeat,
                )
                for train_idx, test_idx in kfold.split(x):
                    model = make_pipeline(
                        PolynomialFeatures(degree=degree),
                        StandardScaler(),
                        Ridge(alpha=0),
                    )
                    model.fit(x[train_idx], y[train_idx].ravel())
                    predictions[test_idx, repeat] = model.predict(x[test_idx]).ravel()

            mean_prediction = np.mean(predictions, axis=1, keepdims=True)
            test_error.append(np.mean((y - predictions) ** 2))
            bias.append(np.mean((y - mean_prediction) ** 2))
            variance.append(np.mean(np.var(predictions, axis=1)))

        decomposition[folds] = (test_error, bias, variance)
        best_idx = int(np.argmin(test_error))
        best_degree = int(degrees[best_idx])
        optimal_parameters[folds] = {"degree": best_degree, "lambda": None, "mse": float(test_error[best_idx])}
        print(
            f"Best degree for {folds}-fold CV: {best_degree}, "
            f"MSE={min(test_error):.6g}"
        )

        plt.figure(figsize=(6.6, 5.0))
        plt.plot(degrees, test_error, "o-", color=colors[0], label="test error")
        plt.plot(
            degrees,
            bias,
            "s-",
            color=colors[1],
            label=r"bias$^2$ (+ $\sigma^2$)",
        )
        plt.plot(degrees, variance, "d-", color=colors[2], label="variance")
        plt.yscale("log")
        plt.title(f"OLS MSE Decomposition ({folds}-Fold CV)")
        plt.xlabel("Polynomial degree", fontsize=14)
        plt.ylabel("MSE decomposition", fontsize=14)
        plt.legend(loc="upper left", fontsize=13)
        plt.xticks(degrees, size=13)
        plt.yticks(size=14)
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"figures/MSE_decomposition_OLS_{folds}fold.pdf", dpi=300)
        plt.close()

    plt.figure(figsize=(8, 5.5))
    for idx, folds in enumerate(folds_list):
        plt.plot(
            degrees,
            decomposition[folds][0],
            "o-",
            label=f"{folds}-fold CV",
            color=colors[idx],
        )
    plt.xlabel("Polynomial degree", fontsize=16)
    plt.ylabel("MSE", fontsize=16)
    plt.yscale("log")
    plt.xticks(degrees, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig("figures/model_selection_OLS.pdf", dpi=300)
    plt.close()
    return optimal_parameters

