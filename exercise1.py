import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error
from sklearn.utils import resample
from numpy import random
#============================== MAIN =================================================

def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

def design_matrix(x, degree, intercept = False):
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
sigma = 1                              # noise level: explore it!
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

x = np.ravel(x)
y = np.ravel(y)

# Defining the Singular Value Decomposition function
def ols(X, y):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ ((U.T @ y) / s)


# Center the target to zero mean
y_mean = y.mean()
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

# plt.figure()
# plt.plot(degrees, mse_train_values, label = "Train mse")
# plt.plot(degrees, mse_test_values, label = "Test mse")
# plt.plot(degrees, R2_values, label = "R2 values")
# plt.legend()
# plt.show()


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

# for i in range(len(degrees)-1):
#     plt.plot(degrees[i:], filtered_theta[i], label=f"Degree {i+1}")

# plt.plot(degrees[14:], filtered_theta[14], "o", label="Degree 15")

# plt.legend()
# plt.grid()
# plt.show()


# Varying the coefficient in front of the added stochastic noise
# print("Varying noise level:")
# for sigma in [0.0, 0.01, 0.1, 0.5, 1.0, 20, 5.0]:

#     x = rng.random((100, 1))

#     x = np.ravel(x)
#     y = np.ravel(y)

#     X = design_matrix(x, degree = 2)
#     y_tilde = X @ ols(X, y)

#     print(f"{sigma} {sigma**2} {mean_squared_error(y, y_tilde)} {r2_score(y, y_tilde)}")


print("Theta values for OLS:")
print(theta_values[:,4])
#=======================================================================================================
# Part b)

degree = 5

def ridge(x,y,lam,degree):
    X = rescale_design_matrix(x, degree)
    return ols(X,y) * (1/(1+lam))

ridge_values = []
lambda_values = np.logspace(-4, 2, 100)
for each in lambda_values:
    ridge_values.append(ridge(x,y,each,degree))

ridge_temp = np.array(ridge_values)

ridge_values=ridge_temp.T

plt.figure()
for i in np.arange(degree):
    plt.plot(lambda_values, ridge_values[i], label=f"Theta={i+1}")
plt.xscale("log")
plt.axhline(y=0, color = "gray", alpha = 0.2)
plt.legend()
plt.show()

print("Theta values for Ridge:")
for each in ridge_values:
    print(each)

# vi sammenlikner start theta-verdier for OLS og Ridge for å sjekke at det gir mening, vi burde starte på ca samme sted siden vi begynner med en liter verdi for lambda. 
# deretter skal vi se hvordan theta verdiene for ridge utvikler seg, spesielt med tanke på hvordan de avhenger av lambda. 


#=======================================================================================================
# Part c)

mse_values=np.array([])

# def calculate_mse(degree):
#     X=design_matrix(x, degree)
#     X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
#     model = sklearn.linear_model.LinearRegression(fit_intercept=False)
#     model.fit(X_train, y_train)
#     mean_squared_error_train = sklearn.metrics.mean_squared_error(y_train, model.predict(X_train))
#     mean_squared_error_test = sklearn.metrics.mean_squared_error(y_test, model.predict(X_test))
#     return mean_squared_error_train, mean_squared_error_test

# degrees = np.arange(1, 16)
mse_train_values_c = []
mse_test_values_c = []

for degree in degrees:
    mse_train_c, mse_test_c, theta_values_c, y_tilde_c = calculate_mse(x, y, degree)
    mse_train_values_c.append(mse_train_c)
    mse_test_values_c.append(mse_test_c)

plt.plot(degrees, mse_train_values_c, label='Train MSE')
plt.plot(degrees, mse_test_values_c, label='Test MSE')
plt.xlabel('Polynomial Degree')
plt.xticks(range(1, 16))
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs Polynomial Degree')
plt.legend()
plt.grid()
plt.show()

print(x.shape)
print(y.shape)


#del 2

BLUE, RED, YELLOW = "#004488", "#BB5566", "#DDAA33" 

x = x.reshape(-1,1)
y = y.reshape(-1,1) 


number_of_bootstraps, polynomial_degree = 100, 16
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=2026)

test_error = []
bias = []
variance = []


for degree in np.arange(polynomial_degree):
    model = make_pipeline(PolynomialFeatures(degree = degree), StandardScaler(), Ridge(alpha = 0)) #this is what actually makes the fit
    
    y_pred = np.zeros((y_test.shape[0], number_of_bootstraps)) #prepare an array of zeros to fill with predicted y_values, for each bootstrap we perform
    for i in range(number_of_bootstraps):
        x_resampled, y_resampled = resample(x_train, y_train)
        y_pred[:, i] = model.fit(x_resampled, y_resampled).predict(x_test).ravel() #this for loop is repeated for each polynomial degree up to the one we have set as a maxmimum
        #it fills all y values for the bootstrap iteration given by i, and saves it for all iterations

    test_error.append(np.mean(np.mean((y_test - y_pred)**2, axis=1, keepdims=True)))
    bias.append(np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True))**2))
    variance.append(np.mean(np.var(y_pred, axis=1, keepdims=True)))    

# plt.figure(figsize=(6.6, 4.0))
plt.plot(np.arange(polynomial_degree), test_error, "o-", color = BLUE, label="test error")
plt.plot(np.arange(polynomial_degree), bias, "s-", color = RED, label=r"bias$^2$ (+ $\sigma^2$)")
plt.plot(np.arange(polynomial_degree), variance, "d-", color = YELLOW, label="variance")

plt.yscale("log")
plt.xlabel("polynomial degree")
plt.ylabel("MSE decomposition")
# plt.xlim(-1, 14)
# plt.ylim(3*10**-3, 3*10**1)
plt.legend(loc = "upper left")
plt.title(f"number of bootstraps = {number_of_bootstraps}")
plt.show()


print(f"For n = {number_of_bootstraps}:")
for each in np.arange(polynomial_degree):
    print(f"For degree", each, f": bias^2 = {bias[each]:.4f}, error = {test_error[each]:.4f}, variance = {variance[each]:.4f}, sum = {(bias[each]+variance[each]):.4f}")

print("The optimal polynomial degree, where the test error is at a minimum, is a polynmial degree of", np.where(test_error == np.min(test_error))[0])



#=======================================================================================================
# Part d)

for k in [5,10]:
    lam = 0
    n, polynomial_degree = 100, 16
    kfold = KFold(n_splits = k, shuffle=True, random_state = 2026)

    mse = np.zeros(polynomial_degree)

    for each in np.arange(polynomial_degree):
        model = make_pipeline(PolynomialFeatures(degree = each), StandardScaler(), Ridge(alpha = lam)) #this is what actually makes the fit
        scores = -cross_val_score(model, x, y, cv = kfold, scoring="neg_mean_squared_error")
        mse[each] = scores.mean()

    print(mse)
    print("The optimal lambda value is", np.arange(polynomial_degree)[np.where(mse == np.min(mse))])

    plt.plot(np.arange(polynomial_degree), mse, "o-", label = f"k-fold = {k}")
    plt.xlabel("polynomial degree")
    plt.yscale("log")
    plt.ylabel("MSE")
    plt.legend()


plt.show()

#=======================================================================================================
# Part e)




    