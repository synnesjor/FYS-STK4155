import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from src.utils import *

# Script for part 1a

def main(x, y):
    # Center the target to zero mean
    y = y - y.mean()

    colours = ["#aa333c", "#99d2fb", "#9aab64", "#21534c", "#ef9d2c", "#B0a6df"]

    degrees = np.arange(1, 16)
    mse_train_values = []
    mse_test_values = []
    theta_values = []
    R2_values_test = []
    R2_values_train = []

    for degree in degrees:
        mse_train, mse_test, theta, y_tilde, y_tilde_test, y_tilde_train, y_train, y_test = calculate_mse(x, y, degree)
        mse_train_values.append(mse_train)
        mse_test_values.append(mse_test)
        theta_values.append(theta)
        R2_values_test.append(r2_score(y_test, y_tilde_test))
        R2_values_train.append(r2_score(y_train, y_tilde_train))

    print("Training data:")
    print("The optimal polynomial degree, according to the R2 value for the training data, is", np.argmax(R2_values_train))
    print("The optimal polynomial degree, according to the MSE value for the training data, is", np.argmin(mse_train_values))

    print("Test data:")
    print("The optimal polynomial degree, according to the R2 value for the test data, is", np.argmax(R2_values_test))
    print("The optimal polynomial degree, according to the MSE value for the test data, is", np.argmin(mse_test_values))

    plt.figure(figsize = (9,6.5))
    plt.plot(degrees, mse_train_values, "-o", label = "Train MSE", color = colours[0], markersize = 3)
    plt.plot(degrees, mse_test_values, "-o", label = "Test MSE", color = colours[1], markersize = 3)
    plt.plot(degrees, R2_values_train, "-o", label = f"$R^2$ train", color = colours[4], markersize = 3)
    plt.plot(degrees, R2_values_test, "-o", label = f"$R^2$ test", color = colours[3], markersize = 3)

    plt.xlabel(f"Polynomial degree", fontsize = 20)
    # plt.ylabel("Mean Square Error (MSE)")
    plt.xticks(size = 18)
    plt.yticks(size = 18)
    plt.grid(alpha = 0.4)
    plt.legend(loc = "center right", fontsize = 20)
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

    plt.figure(figsize = (9.5,7))
    for i in range(len(degrees)-1):
        plt.plot(degrees[i:], filtered_theta[i], label=f"$\\theta_{{{(i+1)}}}$")

    plt.plot(degrees[14:], filtered_theta[14], "o", label="$\\theta_{15}$")
    plt.xlabel(f"Polynomial degree", fontsize = 22)
    plt.ylabel(f"Polynomial coefficient $\\theta$", fontsize = 22)
    plt.xticks(size = 18)
    plt.yticks(size = 18)
    plt.grid(alpha = 0.4)
    plt.legend(fontsize = 16)
    plt.savefig('figures/theta_vs_degree.pdf', dpi=300)

    print("Theta values for OLS:")
    print(theta_values[:,4])

