import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from src.utils import *

# Script for part 1a

def main(x, y):

    # Defining the Singular Value Decomposition function
    def ols(X, y):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        return Vt.T @ ((U.T @ y) / s)

    # Center the target to zero mean
    y = y - y.mean()

    mse_values=np.array([])

    def calculate_mse(x, y, degree):
        X = rescale_design_matrix(x, degree)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 2026)
        model = LinearRegression(fit_intercept=False)
        model.fit(X_train, y_train)
        y_tilde = model.predict(X)
        mean_squared_error_train = mean_squared_error(y_train, model.predict(X_train))
        mean_squared_error_test = mean_squared_error(y_test, model.predict(X_test))
        theta = ols(X, y)
        return mean_squared_error_train, mean_squared_error_test, theta, y_tilde

    degrees = np.arange(1, 16)
    mse_train_values = []
    mse_test_values = []
    theta_values = []
    R2_values = []

    for degree in degrees:
        mse_train, mse_test, theta, y_tilde = calculate_mse(x, y, degree)
        mse_train_values.append(mse_train)
        mse_test_values.append(mse_test)
        theta_values.append(theta)
        R2_values.append(r2_score(y, y_tilde))

    plt.figure()
    plt.plot(degrees, mse_train_values, label = "Train mse")
    plt.plot(degrees, mse_test_values, label = "Test mse")
    plt.plot(degrees, R2_values, label = "R2 values")
    plt.legend()
    plt.savefig('figures/mse_values.pdf', dpi=300)

    
    theta_list=list(theta_values)
    for j in range(len(theta_list)):
        for i in range(len(theta_list[-1])-j-1):
            theta_list[j]=np.append(theta_list[j], np.nan) # Using NaN to avoid removing parameter values...

    theta_values=np.array(theta_list)
    theta_values=theta_values.T

    filtered_theta = []

    for i in range(len(theta_values)):
        mask = np.isnan(theta_values[i]) == False
        filtered_data = theta_values[i][mask]
        filtered_theta.append(filtered_data)

    print("Theta values for OLS:")
    print(theta_values[:,4])