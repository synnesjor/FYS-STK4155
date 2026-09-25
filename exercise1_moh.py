mse_values=np.array([])

def calculate_mse(degree):
    X=design_matrix(x, degree)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
    model = sklearn.linear_model.LinearRegression(fit_intercept=False)
    model.fit(X_train, y_train)
    mean_squared_error_train = sklearn.metrics.mean_squared_error(y_train, model.predict(X_train))
    mean_squared_error_test = sklearn.metrics.mean_squared_error(y_test, model.predict(X_test))
    return mean_squared_error_train, mean_squared_error_test

degrees=np.arange(1, 16)
mse_train_values = []
mse_test_values = []

for degree in degrees:
    mse_train, mse_test = calculate_mse(degree)
    mse_train_values.append(mse_train)
    mse_test_values.append(mse_test)

plt.plot(degrees, mse_train_values, label='Train MSE')
plt.plot(degrees, mse_test_values, label='Test MSE')
plt.xlabel('Polynomial Degree')
plt.xticks(range(1, 16))
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error vs Polynomial Degree')
plt.legend()
plt.grid()
plt.show()