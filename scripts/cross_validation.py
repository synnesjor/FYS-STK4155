import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score
from src.utils import *

# Script for part 1d

def main(x, y):
    x = x.reshape(-1,1)
    y = y.reshape(-1,1) 

    n = 100

    colours = ["#aa333c", "#99d2fb", "#9aab64", "#21534c", "#ef9d2c", "#B0a6df"]

    plt.figure(figsize = (8,5.5))
    for idx, k in enumerate([5,10, n]):
        lam = 0
        polynomial_degree = 16
        kfold = KFold(n_splits = k, shuffle=True, random_state = 2026)

        mse = np.zeros(polynomial_degree)

        for each in np.arange(polynomial_degree):
            model = make_pipeline(PolynomialFeatures(degree = each), StandardScaler(), Ridge(alpha = lam)) #this is what actually makes the fit
            print(each, model)
            scores = -cross_val_score(model, x, y, cv = kfold, scoring="neg_mean_squared_error")
            mse[each] = scores.mean()

        print("The optimal polynomial degree is", np.arange(polynomial_degree)[np.where(mse == np.min(mse))])
        plt.plot(np.arange(polynomial_degree), mse, "o-", label = f"k-fold = {k}", color = colours[idx])
        plt.xlabel("polynomial degree", fontsize = 16)
        plt.yscale("log")
        plt.ylabel("MSE", fontsize = 16)
        plt.tick_params(axis='x', labelsize=16)
        plt.tick_params(axis='y', labelsize=16)
        plt.gca().yaxis.get_offset_text().set_fontsize(16)
        plt.legend(fontsize = "14")
        plt.grid(alpha = 0.4)
        plt.savefig('figures/cross_validation_ols.pdf', dpi=300)


    nlambdas = 100
    our_logspace = np.logspace(-5,0,n)
    k = [5,10,n]

    optimal_lambdas = []

    plt.figure(figsize = (8.5,6))
    for i, each in enumerate(k):
        cv_mse = np.zeros(len(our_logspace))
        for idx, lam in enumerate(our_logspace):
            kfold = KFold(n_splits = each, shuffle=True, random_state=2026)
            model = make_pipeline(PolynomialFeatures(degree = 6), StandardScaler(), Ridge(alpha = lam)) #this is what actually makes the fit
            scores = -cross_val_score(model, x, y, cv = kfold, scoring="neg_mean_squared_error")
            cv_mse[idx] = np.mean(scores)

        optimal_lambdas.append(our_logspace[np.where(cv_mse == np.min(cv_mse))][0])

        plt.plot((our_logspace), cv_mse, "--", color = colours[i], label = f"k-fold = {each}")
        plt.plot((our_logspace[np.where(cv_mse == np.min(cv_mse))][0]), np.min(cv_mse), "o", color = colours[i+3], label = rf"optimal $\lambda$ for k-fold {each}")
        plt.xscale("log")
        plt.xlabel(rf"penalty parameter $\lambda$", fontsize = "18")
        plt.ylabel("MSE", fontsize = "18")
        plt.xticks(size = "16")
        plt.yticks(size = "16")
        plt.legend(fontsize = 16)
        plt.grid(alpha = 0.4)
        print("The optimal lambda (minimum MSE) is a lambda value of", our_logspace[np.where(cv_mse == np.min(cv_mse))][0])
    plt.savefig(f"figures/cross_validation_ridge.pdf")

        # print(np.where(cv_mse == np.min(cv_mse)))



                
        