import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.utils import resample
from sympy import degree
from src.utils import *
from scripts import model_selection_ols, model_selection_ridge, model_selection_lasso

def main(x, y, y_offset=0.0):

    print("Running model selection for OLS, Ridge, and Lasso...")
    print("OLS")
    ols_parameters = model_selection_ols.main(x, y)
    print("Ridge")
    ridge_parameters = model_selection_ridge.main(x, y)
    print("Lasso")
    lasso_parameters = model_selection_lasso.main(x, y)
    
    x=x.reshape(-1,1)
    y=y.reshape(-1,1)

    x_plot = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
    y_runge = 1.0 / (1.0 + 25.0 * x_plot**2) - y_offset

    print("Plotting estimated Runge function for OLS, Ridge, and Lasso...")
    # OLS
    plt.figure(figsize=(8, 5.5))
    plt.scatter(x, y, color="#21534c", s=24, alpha=0.7, label="Observed data")
    plt.plot(
        x_plot,
        y_runge,
        color="#21534c",
        linestyle="--",
        linewidth=2,
        label="Centered Runge function",
    )

    colors = ("#aa333c", "#ef9d2c", "#99d2fb")
    for (folds, parameters), color in zip(sorted(ols_parameters.items()), colors):
        degree = parameters["degree"]
        model_ols = make_pipeline(PolynomialFeatures(degree=degree), StandardScaler(), LinearRegression())
        model_ols.fit(x, y.ravel())
        plt.plot(x_plot, model_ols.predict(x_plot), color=color, linewidth=2, label=f"OLS: {folds}-fold CV (degree {degree})")

    plt.xlabel("x")
    plt.ylabel("Centered y")
    plt.title("OLS Estimates of the Runge Function")
    plt.grid(alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/estimated_runge_ols.pdf", dpi=300)


    # Ridge
    plt.figure(figsize=(8, 5.5))
    plt.scatter(x, y, color="#21534c", s=24, alpha=0.7, label="Observed data")
    plt.plot(x_plot, y_runge, color="#21534c", linestyle="--", linewidth=2, label="Centered Runge function")
    for (folds, parameters), color in zip(sorted(ridge_parameters.items()), colors):
        degree = parameters["degree"]
        alpha = parameters["lambda"]
        model_ridge = make_pipeline(PolynomialFeatures(degree=degree), StandardScaler(), Ridge(alpha=alpha))
        model_ridge.fit(x, y.ravel())
        plt.plot(x_plot, model_ridge.predict(x_plot), color=color, linewidth=2, label=f"Ridge: {folds}-fold CV (degree {degree}, lambda={alpha:.3g})")

    plt.xlabel("x")
    plt.ylabel("Centered y")
    plt.title("Ridge Estimates of the Runge Function")
    plt.grid(alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/estimated_runge_ridge.pdf", dpi=300)


    # Lasso
    plt.figure(figsize=(8, 5.5))
    plt.scatter(x, y, color="#21534c", s=24, alpha=0.7, label="Observed data")
    plt.plot(x_plot, y_runge, color="#21534c", linestyle="--", linewidth=2, label="Centered Runge function")
    for (folds, parameters), color in zip(sorted(lasso_parameters.items()), colors):
        degree = parameters["degree"]
        alpha = parameters["lambda"]
        model_lasso = make_pipeline(PolynomialFeatures(degree=degree), StandardScaler(), Lasso(alpha=alpha, max_iter=1000000, tol=1e-2, selection="random", random_state=2026))
        model_lasso.fit(x, y.ravel())
        plt.plot(x_plot, model_lasso.predict(x_plot), color=color, linewidth=2, label=f"Lasso: {folds}-fold CV (degree {degree}, lambda={alpha:.3g})")
    plt.xlabel("x")
    plt.ylabel("Centered y")
    plt.title("Lasso Estimate of the Runge Function")
    plt.grid(alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/estimated_runge_lasso.pdf", dpi=300)

