import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from src.utils import *

# Script for part 1c
def main(x, y):
    BLUE, RED, YELLOW = "#004488", "#BB5566", "#DDAA33" 

    degrees = np.arange(1, 16)
    
    x = x.reshape(-1,1)
    y = y.reshape(-1,1) 

    number_of_bootstraps = 100
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=2026)

    test_error = []
    bias = []
    variance = []

    for degree in degrees:
        model = make_pipeline(PolynomialFeatures(degree = degree), StandardScaler(), Ridge(alpha = 0)) #this is what actually makes the fit
        
        y_pred = np.zeros((y_test.shape[0], number_of_bootstraps)) #prepare an array of zeros to fill with predicted y_values, for each bootstrap we perform
        for i in range(number_of_bootstraps):
            x_resampled, y_resampled = resample(x_train, y_train)
            y_pred[:, i] = model.fit(x_resampled, y_resampled).predict(x_test).ravel() #this for loop is repeated for each polynomial degree up to the one we have set as a maxmimum
            #it fills all y values for the bootstrap iteration given by i, and saves it for all iterations

        test_error.append(np.mean(np.mean((y_test - y_pred)**2, axis=1, keepdims=True)))
        bias.append(np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True))**2))
        variance.append(np.mean(np.var(y_pred, axis=1, keepdims=True)))    

    plt.figure(figsize=(6.6, 4.0))
    plt.plot(degrees, test_error, "o-", color = BLUE, label="test error")
    plt.plot(degrees, bias, "s-", color = RED, label=r"bias$^2$ (+ $\sigma^2$)")
    plt.plot(degrees, variance, "d-", color = YELLOW, label="variance")

    plt.yscale("log")
    plt.xlabel("polynomial degree")
    plt.ylabel("MSE decomposition")
    # plt.xlim(-1, 14)
    # plt.ylim(3*10**-3, 3*10**1)
    plt.legend(loc = "upper left")
    plt.title(f"number of bootstraps = {number_of_bootstraps}")
    plt.savefig('figures/MSE_decomposition.pdf', dpi=300)


    print(f"For n = {number_of_bootstraps}:")
    for each in degrees:
        print(f"For degree", each, f": bias^2 = {bias[each-1]:.4f}, error = {test_error[each-1]:.4f}, variance = {variance[each-1]:.4f}, sum = {(bias[each-1]+variance[each-1]):.4f}")

    print("The optimal polynomial degree, where the test error is at a minimum, is a polynmial degree of", np.where(test_error == np.min(test_error))[0]+1) # Degree is one above the index
