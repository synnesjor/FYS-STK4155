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

    for k in [5,10]:
        lam = 0
        polynomial_degree = 16
        kfold = KFold(n_splits = k, shuffle=True, random_state = 2026)

        mse = np.zeros(polynomial_degree)

        for each in np.arange(polynomial_degree):
            model = make_pipeline(PolynomialFeatures(degree = each), StandardScaler(), Ridge(alpha = lam)) #this is what actually makes the fit
            scores = -cross_val_score(model, x, y, cv = kfold, scoring="neg_mean_squared_error")
            mse[each] = scores.mean()

        print("The optimal lambda value is", np.arange(polynomial_degree)[np.where(mse == np.min(mse))])

        plt.plot(np.arange(polynomial_degree), mse, "o-", label = f"k-fold = {k}")
        plt.xlabel("polynomial degree")
        plt.yscale("log")
        plt.ylabel("MSE")
        plt.legend()
        plt.savefig('figures/cross_validation.pdf', dpi=300)