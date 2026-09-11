import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#============================== MAIN =================================================

def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

def design_matrix(x, degree, intercept = False):
    # polynomial features [1, x, x^2, ..., x^degree] (drop the 1 if intercept=False)
    start = 0 if intercept else 1
    return np.vstack([x**p for p in range(start, degree + 1)]).T

def rescale_design_matrix(x, degree):
    X = design_matrix(x, degree)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1  # safeguard to avoid division by zero for constant features
    X_norm = (X - X_mean) / X_std
    return X_norm

rng = np.random.default_rng(2026)
n = 100
sigma = 0.1                                  # noise level: explore it!
x = np.sort(rng.uniform(-1, 1, n))
y = runge(x) + rng.normal(0, sigma, n) #full function

xx = np.linspace(-1, 1, 400)
# plt.figure()
# plt.plot(xx, runge(xx), color="#004488", label="Runge's function")
# plt.scatter(x, y, s=12, color="#BB5566", label=rf"data, $\sigma={sigma}$")
# plt.xlabel("x"); plt.ylabel("y"); plt.legend(frameon=False)
# plt.show()

#===============================================================================

# part a)

# print(rescale_design_matrix(x,2))

x = np.ravel(x)
y = np.ravel(y)

# Defining the Singular Value Decomposition function
def ols(X, y):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ ((U.T @ y) / s)


# Center the target to zero mean
y_mean = y.mean()
y = y - y.mean()

# X = rescale_design_matrix(x, degree = 2) # shape (100,3)
# theta = ols(X, y)
# print(theta)


# print(f"2.1) Optimal parameters from the SVD:" , theta)

# Code for 2.2
# Now using sklearn
# reg = LinearRegression(fit_intercept=False)
# reg_fit = reg.fit(X, y)
# theta_sklearn = reg_fit.coef_

# print("2.2) Optimal parameters from sklearn:", theta_sklearn)

# Code for 2.3
# Models predicted values
# y_tilde = X @ theta_sklearn

# Compute the mean square error using sklearn
mse_values=np.array([])

def calculate_mse(x, y, degree):
    X = rescale_design_matrix(x, degree)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 2026)
    model = LinearRegression(fit_intercept=False)
    model.fit(X_train, y_train)
    y_tilde = model.predict(X)
    mean_squared_error_train = mean_squared_error(y_train, model.predict(X_train))
    mean_squared_error_test = mean_squared_error(y_test, model.predict(X_test))
    theta = ols(X, y) # shape (3,1) #there is an issue here
    return mean_squared_error_train, mean_squared_error_test, theta, y_tilde



degrees = np.arange(1, 4)
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

print(theta_values)
# plt.figure()
# plt.plot(degrees, mse_train_values, label = "Train mse")
# plt.plot(degrees, mse_test_values, label = "Test mse")
# plt.plot(degrees, R2_values, label = "R2 values")
# plt.legend()
# plt.show()



plt.figure()
for idx, i in enumerate(theta_values):
    if idx == 0: 
        plt.plot(0,i, "o", label = "Degree 1")
    else: plt.plot(np.arange(len(i)), i, label = f"Degree {idx+1}")


plt.legend()
plt.show()

# And the R2 score function

# Varying the coefficient in front of the added stochastic noise
print("Varying noise level:")
for sigma in [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:

    # rng = np.random.default_rng(2026)
    x = rng.random((100, 1))

    x = np.ravel(x)
    y = np.ravel(y)

    X = design_matrix(x, degree = 2)
    y_tilde = X @ ols(X, y)

    print(f"{sigma} {sigma**2} {mean_squared_error(y, y_tilde)} {r2_score(y, y_tilde)}")
