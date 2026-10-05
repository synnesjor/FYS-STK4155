""" 
    LLM ASSISTED: 
    
    Tools: Chat GPT and UiO GPT (September / October 2026) 
    Role: The LLM has fixed bugs and written short snippets of code to help with formatting plots.
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.utils import resample
from src.utils import *

def main(x, y):

    x = x.reshape(-1,1)
    y = y.reshape(-1,1) 

    # refer to cross_validation_ols.pdf for selection of OLS
    polynomial_degree = 16

    nlambdas = 100
    our_logspace = np.logspace(-5,0,nlambdas)
    k = [5, 10, 25]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=2026
    )

    optimal_lambdas = {folds: [] for folds in k}

    for degree in range(1, polynomial_degree+1):
        for folds in k:
            cv_mse = np.zeros(len(our_logspace))
            for idx, lam in enumerate(our_logspace):
                kfold = KFold(n_splits=folds, shuffle=True, random_state=2026)
                model = make_pipeline(PolynomialFeatures(degree=degree), StandardScaler(), Ridge(alpha=lam))
                scores = -cross_val_score(model, x_train, y_train.ravel(), cv=kfold, scoring="neg_mean_squared_error")
                cv_mse[idx] = scores.mean()

            optimal_lambdas[folds].append(our_logspace[np.argmin(cv_mse)])
            print(f"Optimal lambda for degree {degree} with {folds}-fold CV: {optimal_lambdas[folds][-1]}")

    degrees = np.arange(1, polynomial_degree + 1)
    number_of_bootstraps = 100
    colors = ["#aa333c", "#99d2fb", "#9aab64"]
    optimal_parameters = {}

    for folds in k:
        test_error = []
        bias = []
        variance = []

        for degree_idx, degree in enumerate(degrees):
            lam = optimal_lambdas[folds][degree_idx]
            model = make_pipeline(PolynomialFeatures(degree=degree), StandardScaler(), Ridge(alpha=lam))
            y_pred = np.zeros((y_test.shape[0], number_of_bootstraps))

            for bootstrap_idx in range(number_of_bootstraps):
                x_resampled, y_resampled = resample(x_train, y_train, random_state=2026 + bootstrap_idx)
                y_pred[:, bootstrap_idx] = model.fit(x_resampled, y_resampled).predict(x_test).ravel()

            test_error.append(np.mean((y_test - y_pred) ** 2))
            mean_prediction = np.mean(y_pred, axis=1, keepdims=True)
            bias.append(np.mean((y_test - mean_prediction) ** 2))
            variance.append(np.mean(np.var(y_pred, axis=1, keepdims=True)))

        best_idx = int(np.argmin(test_error))
        optimal_parameters[folds] = {"degree": int(degrees[best_idx]), "lambda": float(optimal_lambdas[folds][best_idx]), "mse": float(test_error[best_idx])}
        print(
            f"Best degree for {folds}-fold CV: {degrees[best_idx]}, "
            f"lambda={optimal_lambdas[folds][best_idx]:.10g}, "
            f"MSE={test_error[best_idx]:.6g}"
        )

        plt.figure(figsize=(6.6, 5.0))
        plt.plot(degrees, test_error, "o-", color=colors[0], label="test error")
        plt.plot(degrees, bias, "s-", color=colors[1], label=r"bias$^2$ (+ $\sigma^2$)",)
        plt.plot(degrees, variance, "d-", color=colors[2], label="variance")
        plt.yscale("log")
        plt.title(f"Ridge MSE Decomposition ({folds}-Fold CV)")
        plt.xlabel("Polynomial degree", fontsize=14)
        plt.ylabel("MSE decomposition", fontsize=14)
        plt.legend(loc="upper left", fontsize=13)
        plt.xticks(degrees, size=13)
        plt.yticks(size=14)
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"figures/MSE_decomposition_ridge_{folds}fold.pdf", dpi=300)
        plt.close()
    return optimal_parameters