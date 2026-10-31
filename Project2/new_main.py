from sklearn.model_selection import train_test_split
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
jax.config.update("jax_enable_x64", True)
import time
from IPython.display import Image
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score, cross_validate, GridSearchCV, validation_curve

def runge_1d(x, sigma, n):
    noise = rng.normal(0, sigma, n).reshape(-1, 1)
    return (1 / (1+(25*x**2))) + noise

def inputs_scale(x):
    X = x
    X_mean = X.mean()
    X_std = X.std()
    # X_std[X_std == 0] = 1  # safeguard to avoid division by zero for constant features
    X_norm = (X - X_mean) / X_std
    return X_norm

SEED = 2026
rng = np.random.default_rng(SEED) #LLM tag: ChatGPT suggested to make this more global, as we were having some issues with reproducibility

n = 100
sigma = 0.1

inputs = rng.uniform(-1,1,n).reshape(-1, 1)
inputs_scaled = inputs_scale(inputs)

targets = runge_1d(inputs_scaled, sigma, n)
targets_centered = targets - np.mean(targets)

X = inputs_scaled
y = targets_centered

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=2026)   # step 1: 20 points set aside
cv = KFold(n_splits=4, shuffle=True, random_state=2026)     # step 2: the 80 non-test points, four folds of 20
print("non-test points:", len(y_train), "  test points:", len(y_test))
for j, (tr, val) in enumerate(cv.split(X_train)):
    print(f"fit {j + 1}: train on {len(tr)} points (folds {[i + 1 for i in range(4) if i != j]}), validate on fold {j + 1} "
          f"({len(val)} points, x from {X_train[val].min():+.3f} to {X_train[val].max():+.3f})")
print(f"the non-test x range: {X_train.min():+.3f} to {X_train.max():+.3f}; the test x range: {X_test.min():+.3f} to {X_test.max():+.3f}")