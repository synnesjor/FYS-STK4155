
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
jax.config.update("jax_enable_x64", True)


#-------------------------------------------------------- FUNCTIONS PART -----------------------------------------------------------------------------
SEED = 2026
rng = np.random.default_rng(SEED) #LLM tag: ChatGPT suggested to make this more global, as we were having some issues with reproducibility


def predict_network(X, layers, activation_functions):
    return feed_forward_batch(X, layers, activation_functions)


def runge_1d(x, sigma, n):
    noise = rng.normal(0, sigma, n).reshape(-1, 1)
    return (1 / (1+(25*x**2))) + noise

def runge_1d_no_noise(x):
    return (1 / (1+(25*x**2)))



def no_activation_func(z):
    return z


def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))


def ReLU(z):
    return jnp.where(z > 0, z, 0.0)


def leaky_ReLU(z, alpha = 0.01):
    return jnp.where(z > 0, z, alpha*z)


def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data: one row of scores per sample."""
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=False))
    return e_z / jnp.sum(e_z, axis=1, keepdims=True)


def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.
    Use this function when you use the activation function on one vector at a time."""
    e_z = jnp.exp(z - jnp.max(z))
    return e_z / jnp.sum(e_z)


def weights(n_in, n_out):
    return np.random.normal(0, 2 / (n_in + n_out))  #her vil vi ha n_in = n_out fra forrige layer, n_out er definert av oss i layer_output_size


def create_layers_batch(network_input_size, layer_output_sizes):
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = rng.normal(0, 1/np.sqrt(i_size), (i_size, layer_output_size))
        # print(np.shape(W))
        b = np.zeros(layer_output_size) + 0.01
        layers.append((W, b))

        i_size = layer_output_size
    return layers


def mse(predict, target):
    return jnp.mean((target-predict)**2)


def cost(input, layers, activation_funcs, target):
    predict = feed_forward_batch(input, layers, activation_funcs)
    return mse(predict, target)


def feed_forward_batch(inputs, layers, activation_funcs):
    a = inputs
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = a @ W + b
        a = activation_func(z)
    return a

def accuracy(predictions, targets):
    """Fraction of rows whose largest predicted probability sits in the correct class."""
    return accuracy_score(np.argmax(np.asarray(predictions), axis=1), np.argmax(np.asarray(targets)))

# i b så ber de om å bruke bare mse - accuracy er ett klassifikasjon problem og vi gjør regression her
def train_network(inputs, layers, activation_funcs, targets, gamma=0.1, epochs = 10000):
    history = []
    for epoch in range(epochs):
        layers_grad = gradient_func(inputs, layers, activation_funcs, targets)
        layers = [(W - gamma*dW, b - gamma*db) for (W, b), (dW, db) in zip(layers, layers_grad)]
        if epoch % 50 == 0 or epoch == epochs - 1:
            predictions = feed_forward_batch(inputs, layers, activation_funcs)
            history.append((epoch, float(cost(inputs, layers, activation_funcs, targets)), mse(predictions, targets)))
    return layers, history, predictions, mse(predictions, targets)

def train_network_gradient(inputs, layers, activation_funcs, targets, gamma=0.01, epochs=10000, method="plain", tol = 0):
    """
    LLM tag:
    With the train_network function as a base, we asked ChatGPT and Claude if it made sense to reuse optimiser step function used in Project 1. 
    When we had our own suggestion for how to combine the functions, we checked with the LLMs if they agreed. We also got the line creating the 
    dictionary for "states", and got some help with debbugging the loop where the layer's weight and biases are updated.     
    """
    # Independent optimizer states for each weight and bias
    states = [({"W": {}, "b": {}}, {"W": {}, "b": {}}) for W, b in layers]
    history = []
    for epoch in range(epochs):
        # param_old = param.copy()
        # Calculate gradients for the whole network
        layers_grad = gradient_func(inputs, layers, activation_funcs, targets)

        new_layers = []
        new_states = []

        # Update each layer's weights and biases
        for (W, b), (dW, db), (state_W, state_b) in zip(layers, layers_grad, states):
            W, state_W = optimiser_step(method, W, dW, state_W,epoch + 1, gamma)
            b, state_b = optimiser_step(method, b, db, state_b,epoch + 1, gamma)
            new_layers.append((W, b))
            new_states.append((state_W, state_b))
        layers = new_layers
        states = new_states

        

        # Record training progress
        if epoch % 50 == 0 or epoch == epochs - 1:
            predictions = feed_forward_batch(inputs, layers, activation_funcs)
            c = cost(inputs, layers, activation_funcs, targets)
            # acc = accuracy(predictions, targets)
            history.append((epoch, float(c), mse(predictions, targets)))

            # if np.abs(c[epoch]-c[epoch-1]) < tol: #this needs to be changed into a proper tolerance definition
            #     print(f"Converged at epoch {epoch}: cost = {c:.6g}")
            #     break
        
        # if np.linalg.norm(param - param_old)>1e2: # breaks if param explodes
        #     break

    return layers, history, predictions, mse(predictions,targets)

def step_length(t, t0, t1):
    """The schedule of Eq. (4.40)."""
    return t0 / (t + t1)

def train_network_stochastic_gradient(inputs, layers, activation_funcs, targets, schedule, gamma=0.01, epochs=10000, method="plain", batch_size=5, tol=1e-6, seed=2026):
    """
    LLM tag:
    When we had our own suggestion for how to combine the functions, we checked with the LLMs if they agreed, especially with the 
    placement of the epochs loop, and updates of t. We also got the convergence check from ChatGPT.
    """
    # Independent optimizer states for each weight and bias
    states = [({"W": {}, "b": {}}, {"W": {}, "b": {}}) for W, b in layers]
    history = []
    t = 0
    for epoch in range(epochs):
        # Calculate gradients for the whole network
        for batch in make_batches(len(inputs), batch_size, rng):
            t += 1
            layers_grad = gradient_func(inputs[batch], layers, activation_funcs, targets[batch])
            gamma_t = gamma if schedule is None else step_length(t, *schedule)    
        
            new_layers = []
            new_states = []

            # Update each layer's weights and biases
            for (W, b), (dW, db), (state_W, state_b) in zip(layers, layers_grad, states):
                W, state_W = optimiser_step(method, W, dW, state_W,epoch + 1, gamma_t)
                b, state_b = optimiser_step(method, b, db, state_b,epoch + 1, gamma_t)
                new_layers.append((W, b))
                new_states.append((state_W, state_b))
            layers = new_layers
            states = new_states

        # Record training progress
        if epoch % 50 == 0 or epoch == epochs - 1: #find out what this does
            predictions = feed_forward_batch(inputs, layers, activation_funcs)
            c = cost(inputs, layers, activation_funcs, targets)
            # acc = accuracy(predictions, targets)
            history.append((epoch, float(c), mse(predictions, targets)))

            if c < tol:
                print(f"Converged at epoch {epoch}: cost = {c:.6g}")
                # break


    return layers, history, predictions, mse(predictions,targets)


def inputs_scale(x):
    X = x
    X_mean = X.mean()
    X_std = X.std()
    # X_std[X_std == 0] = 1  # safeguard to avoid division by zero for constant features
    X_norm = (X - X_mean) / X_std
    return X_norm


def optimiser_step(method, param, g, state, t, gamma, beta=0.9, rho=0.99,
                beta1=0.9, beta2=0.999, eps=1e-6):
    """One update of param from the gradient g at step t (t = 1, 2, ...), Eqs. (4.10), (4.28),
    (4.42)-(4.45), (4.47)-(4.48) and (4.51)-(4.55).  state carries the running quantities."""
    if method == "plain":
        return param - gamma * g, state
    if method == "momentum":
        v = beta * state.get("v", 0.0) + gamma * g               # Eq. (4.28)
        state["v"] = v
        return param - v, state
    if method == "adagrad":
        r = state.get("r", 0.0) + g * g                           # Eq. (4.42)
        state["r"] = r
        return param - gamma * g / (np.sqrt(r) + eps), state      # Eq. (4.45)
    if method == "rmsprop":
        r = rho * state.get("r", 0.0) + (1.0 - rho) * g * g       # Eq. (4.47)
        state["r"] = r
        return param - gamma * g / (np.sqrt(r) + eps), state      # Eq. (4.48)
    if method == "adam":
        m = beta1 * state.get("m", 0.0) + (1.0 - beta1) * g       # Eq. (4.51)
        r = beta2 * state.get("r", 0.0) + (1.0 - beta2) * g * g   # Eq. (4.52)
        state["m"], state["r"] = m, r
        m_hat = m / (1.0 - beta1**t)                              # Eq. (4.54)
        r_hat = r / (1.0 - beta2**t)
        return param - gamma * m_hat / (np.sqrt(r_hat) + eps), state   # Eq. (4.55)
    raise ValueError(f"unknown method {method}")

def gradient_batch(theta, X, y, lam=0.0):
    """Eqs. (4.13) and (4.17): the gradient of (1/n)||X theta - y||^2 + lambda theta^T theta."""
    n = len(y)
    return (2.0 / n) * X.T @ (X @ theta - y) + 2.0 * lam * theta

def make_batches(n, batch_size, rng):
    """Shuffle the indices and split them into minibatches."""
    idx = rng.permutation(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]

def step_length(t, t0, t1):
    """The schedule of Eq. (4.40)."""
    return t0 / (t + t1)


#for OLS comparison
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

def ols(X, y):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ ((U.T @ y) / s)

def calculate_mse(x, y, degree):
    X = rescale_design_matrix(x, degree)
    y = y - np.mean(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 2026)
    model = LinearRegression(fit_intercept=False)
    model.fit(X_train, y_train)
    y_tilde = model.predict(X)
    y_tilde_test = model.predict(X_test)
    y_tilde_train = model.predict(X_train)
    mean_squared_error_train = mean_squared_error(y_train, model.predict(X_train))
    mean_squared_error_test = mean_squared_error(y_test, model.predict(X_test))
    theta = ols(X_test, y_test)
    return mean_squared_error_train, mean_squared_error_test, theta, y_tilde, y_tilde_test, y_tilde_train, y_train, y_test

#-------------------------------------------------------- FUNCTIONS PART -----------------------------------------------------------------------------



#-------------------------------------------------------- GENERAL PART -----------------------------------------------------------------------------
part = "part_c"
sigma = 0.1
n = 2000

inputs = rng.uniform(-1,1,n).reshape(-1, 1)
inputs_scaled = inputs_scale(inputs)

targets = runge_1d(inputs_scaled, sigma, n)
targets_no_noise = runge_1d_no_noise(inputs_scaled)
targets_centered = targets - np.mean(targets)
targets_centered_no_noise = targets_no_noise - np.mean(targets_no_noise)

inputs_scaled, x_test, targets_centered, y_test = train_test_split(inputs_scaled, targets_centered, test_size = 0.2, random_state = 2026)
targets_test = runge_1d(x_test, sigma, len(x_test))

gradient_func = jax.grad(cost, argnums=1)  # Taking the gradient wrt. the second input to the cost function, i.e. the layers
#-------------------------------------------------------- GENERAL PART -----------------------------------------------------------------------------




#-------------------------------------------------------- PART b) -----------------------------------------------------------------------------
if part == "part_b":

    #for two hidden layers with size 100
    network_input_size = inputs_scaled.shape[1]
    layer_output_sizes_100 = [100, 100, 1]
    activation_funcs_100 = [sigmoid, sigmoid, no_activation_func]
    layers_batch_100 = create_layers_batch(network_input_size, layer_output_sizes_100)
    predictions_100 = feed_forward_batch(inputs_scaled, layers_batch_100, activation_funcs_100)
    layers_grad_100 = gradient_func(inputs_scaled, layers_batch_100, activation_funcs_100, targets_centered)
    trained_100, history_100, preds_100, mse_values = train_network(inputs_scaled, layers_batch_100, activation_funcs_100, targets_centered)


    #for one hidden layer with size 50
    layer_output_sizes_50 = [50,1]
    activation_funcs_50 = [sigmoid, no_activation_func]
    layers_batch_50 = create_layers_batch(network_input_size, layer_output_sizes_50)
    predictions = feed_forward_batch(inputs_scaled, layers_batch_50, activation_funcs_50)
    layers_grad_50 = gradient_func(inputs_scaled, layers_batch_50, activation_funcs_50, targets_centered) 
    trained_50, history_50, preds_50, mse_values = train_network(inputs_scaled, layers_batch_50, activation_funcs_50, targets_centered)


    colours = ["#aa333c", "#99d2fb", "#9aab64", "#21534c", "#ef9d2c", "#B0a6df", "#2da9cf", "#ff46a2", "#e5194f", "#8f1cab", "#2da9cf", "#3ba336"]
    learning_rates = [0.001, 0.01, 0.1, 0.3]


   

   

    


#-------------------------------------------------------- PART b) -----------------------------------------------------------------------------





#-------------------------------------------------------- PART c) -----------------------------------------------------------------------------
if part == "part_c":
    ols_deg11_fold25 = np.loadtxt("best_fit_from_project_1_x_and_y.txt")
    print(np.shape(inputs_scaled))
    print(np.shape(targets_centered))
    mse_train, mse_test_ols, theta, y_tilde, y_tilde_test, y_tilde_train, y_train, y_test= calculate_mse(inputs[:,0], targets[:,0], degree=7)
    act_functions = [sigmoid, leaky_ReLU, ReLU]
    act_functions_names = ["sigmoid", "leaky_ReLU", "ReLU"]
    n_hidden_layers = [1,2,3,5]
    # n_hidden_layers = [1] 

    colours = ["#aa333c", "#99d2fb", "#4e827b", "#ef9d2c", "#B0a6df", "#2da9cf", "#ff46a2", "#9aab64", "#e5194f", "#8f1cab", "#2da9cf", "#3ba336"]
    colours2 = ["#ef9d2c", "#020202", "#195626", "#ef2c2c", "#732775", "#a495cd", "#aaaf0f", "#f4b537", "#5513b9", "#5a585b", "#103a46", "#b83453"]

    
    network_input_size = inputs_scaled.shape[1]


#Kode fra Marie med hjelp av Claude med endring IKKE ENDRE
#for plotting the actual fit, adam and rmsprop stochastic:
nodes = [50, 100]
n_hidden_layers = [1, 2]

methods = ["adam", "rmsprop"]

for met in methods:
    if met == "adam":
        gamma_value = 0.013 
    if met == "rmsprop":
        gamma_value = 0.0015
        
    m = 0 #just for colours in plot
    plt.figure(figsize=(8.5, 7.5))
 
    """
    LLM tag:
    The three lines below, used for sorting the points for the true function, was written by ChatGPT.
    """
    points = sorted(zip(inputs_scaled, targets_centered))
    x_sorted = [p[0] for p in points]
    y_sorted = [p[1] for p in points]
    points = sorted(zip(inputs_scaled, targets_centered))
    x_sorted = [p[0] for p in points]
    y_sorted = [p[1] for p in points]
    plt.plot(x_sorted, y_sorted,  "-", color = "#B5B4B4", label = "True function", markersize = 1, alpha = 0.5)
    plt.plot(x_test, y_tilde_test, "o", color = "#ef9d2c", label = "Polyomial fit using OLS, deg 6", markersize = 2)
    plt.plot(ols_deg11_fold25[:,0], ols_deg11_fold25[:,1]+np.mean(targets), "o", color = "#729A72", label = "Polyomial fit using OLS, k-fold 25, deg 11", markersize = 2)

    smallest_mse_values = []
    smallest_mse_values_args = []

    for idx, nod in enumerate(nodes):
        for n in n_hidden_layers:
            if (nod == 50 and n == 1) or (nod == 100 and n == 2):
                if (nod == 50 and n == 1) and met == "adam":
                    gamma_value = 0.013
                if (nod == 50 and n == 1) and met == "adam":
                    gamma_value = 0.013
                if (nod == 50 and n == 1) and met == "rms_prop":
                    gamma_value = 0.0015
                if (nod == 50 and n == 1) and met == "rms_prop":
                    gamma_value = 2.1*10**-3
                mse_values_train = []
                mse_values_test = []
                layer_output_sizes = [nod] * n + [1]
                activation_funcs = [sigmoid] * n + [no_activation_func]
                layers_batch = create_layers_batch(network_input_size,layer_output_sizes)
                print("finding trained")
                trained_layers, history, preds, mse_v = train_network_stochastic_gradient(inputs_scaled, layers_batch, activation_funcs, targets_centered, gamma = gamma_value, method=met, schedule=None, epochs=100)
                test_preds = predict_network(x_test,trained_layers,activation_funcs)
                mse_values_train.append(mse_v)
                test_mse = mse(test_preds, targets_test)
                mse_values_test.append(test_mse)
                # Plot the MSE values for this architecture
                # plt.plot(x_test, test_preds, "-o", markersize = 4, color = colours2[idx+m], label= f"{nod} nodes, {n} hidden layer{'s' if n > 1 else ''}, train")
                plt.plot(x_test, test_preds, "o", markersize = 2, color = colours[idx+m], label=f"{nod} nodes, {n} hidden layer{'s' if n > 1 else ''}, test preds") #LLM tag: Claude helped with the placement of these plot commands

            m += 1

    plt.plot()
    plt.xlabel("x", fontsize=15)
    plt.ylabel("y", fontsize=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.legend(fontsize = 15)
    plt.ylim(-0.33, 1.4)
    plt.savefig(f"figures/fit_to_runge_{met}_ols_7_and_11.pdf")
    plt.show()
    #------------------------------------------------------------------------------------------------------------------------------------------------


    #for c)
    # plt.figure()
    # for idx, a in enumerate(act_functions):
    #     mse_values_train = []
    #     mse_values_test = []
    #     for n in n_hidden_layers:
    #         layer_output_sizes = [100] * n + [1]
    #         activation_funcs = [a] * n + [no_activation_func] #LLM use: Claude used to efficiently create the array for activation functions
    #         print(activation_funcs)
    #         layers_batch = create_layers_batch(network_input_size, layer_output_sizes)
    #         predictions = feed_forward_batch(inputs_scaled, layers_batch, activation_funcs)
    #         layers_grad = gradient_func(inputs_scaled, layers_batch, activation_funcs, targets_centered)
    #         trained_layers, history, preds, mse_v = train_network(inputs_scaled, layers_batch, activation_funcs, targets_centered)
    #         mse_values_train.append(mse_v)
    #         for_test_data = predict_network(x_test, trained_layers, activation_funcs)
    #         mse_for_test_data = mse(for_test_data, targets_test)
    #         mse_values_test.append(mse_for_test_data)
    #         # for epoch, c, mse in history:
    #         #     print(f"epoch {epoch:4d}: cost {c:.4f}  mse {mse:.4f}")
    #     plt.plot(n_hidden_layers, mse_values_train, "-o", color = colours[idx], label = f"{act_functions_names[idx]} train")
    #     plt.plot(n_hidden_layers, mse_values_test, "-o", color = colours[idx+4], label = f"{act_functions_names[idx]} test")
    #     plt.legend()
    # plt.show()

    # nodes = [50,100]
    # n_hidden_layers = [1,2]

    #================================================================================================================================================
    # FOR PLAIN IKKE ENDRE KODEN:
    # "Kode fra Marie med hjelp av Claude med endring"
    # # gamma_values = np.logspace(-3,0,2)
    # gamma_values = np.logspace(-3, np.log10(0.3), 10)
    # print(gamma_values)
    # nodes = [50, 100]
    # n_hidden_layers = [1, 2]

    # m = 0
    # plt.figure(figsize=(8, 5))

    # for idx, nod in enumerate(nodes):
    #     for n in n_hidden_layers:
    #         mse_values_train = []
    #         mse_values_test = [] #LLM tag: Claude helped with the placement of the declariation of these lists
    #         layer_output_sizes = [nod] * n + [1]
    #         activation_funcs = [sigmoid] * n + [no_activation_func]
    #         for gam in gamma_values:
    #             print(f"for gamma = {gam}")
    #             layers_batch = create_layers_batch(network_input_size, layer_output_sizes)
    #             trained_layers, history, preds, mse_v = train_network_gradient(inputs_scaled, layers_batch, activation_funcs, targets_centered, gamma=gam)
    #             test_preds = predict_network(x_test, trained_layers, activation_funcs)
    #             mse_values_train.append(mse_v)
    #             test_mse = mse(test_preds, targets_test)
    #             mse_values_test.append(test_mse)
    #         plt.plot(gamma_values, mse_values_train, "-o", color = colours2[idx+m], label= f" train: {nod} nodes, {n} hidden layer{'s' if n > 1 else ''}")
    #         plt.plot(gamma_values, mse_values_test, "-o", color = colours[idx+m], label=f"test: {nod} nodes, {n} hidden layer{'s' if n > 1 else ''}") #LLM tag: Claude helped with the placement of these plot commands
    #         print("mse train is", mse_values_train)
    #         print("mse test is", mse_values_test)
    #         m+=1
    #     print(np.min(mse_values_train), np.argmin(mse_values_train))
    #     print(np.min(mse_values_test), np.argmin(mse_values_test))

    # plt.xlabel("Learning rate γ")
    # plt.ylabel("MSE")
    # plt.legend()
    # plt.show()
    #================================================================================================================================================



    # #------------------------------------------------------------------------------------------------------------------------------------------------
    # "Kode fra Marie med hjelp av Claude med endring IKKE ENDRE"
    # # gamma_values = np.logspace(-3,0,2)
    # nodes = [50, 100]
    # n_hidden_layers = [1, 2]

    # methods = ["rmsprop"]

    # for met in methods:
    #     if met == "adam":
    #         gamma_values = np.logspace(-3, -1, 10)
    #     if met == "rmsprop":
    #         gamma_values = np.logspace(-3, np.log10(0.03), 10)
    #     print(gamma_values)
    #     m = 0 #just for colours in plot
    #     if met == "adam":
    #         plt.figure(figsize=(8.5, 5.5))
    #     if met == "rmsprop":
    #         plt.figure(figsize=(9, 5.5))

    #     smallest_mse_values = []
    #     smallest_mse_values_args = []
    
    #     for idx, nod in enumerate(nodes):
    #         for n in n_hidden_layers:
    #             if (nod == 50 and n == 1) or (nod == 100 and n == 2):
    #                 mse_values_train = []
    #                 mse_values_test = []
    #                 layer_output_sizes = [nod] * n + [1]
    #                 activation_funcs = [sigmoid] * n + [no_activation_func]
    #                 for gam in gamma_values:
    #                     print(f"for gamma = {gam}")
    #                     layers_batch = create_layers_batch(network_input_size,layer_output_sizes)
    #                     trained_layers, history, preds, mse_v = train_network_stochastic_gradient(inputs_scaled, layers_batch, activation_funcs, targets_centered, gamma=gam, method=met, schedule=None, epochs=100)
    #                     test_preds = predict_network(x_test,trained_layers,activation_funcs)
    #                     mse_values_train.append(mse_v)
    #                     test_mse = mse(test_preds, targets_test)
    #                     mse_values_test.append(test_mse)
    #                 # Plot the MSE values for this architecture
    #                 plt.plot(gamma_values, mse_values_train, "-o", markersize = 4, color = colours2[idx+m], label= f"{nod} nodes, {n} hidden layer{'s' if n > 1 else ''}, train")
    #                 plt.plot(gamma_values, mse_values_test, "-o", markersize = 4, color = colours[idx+m], label=f"{nod} nodes, {n} hidden layer{'s' if n > 1 else ''}, test") #LLM tag: Claude helped with the placement of these plot commands
    #                 print("MSE train is:", mse_values_train)
    #                 print("MSE test is:", mse_values_test)
    #                 # Save the lowest MSE for THIS architecture,
    #                 # plus the index where it occurs in mse_values_test.
    #                 smallest_mse_values.append(np.min(mse_values_test))
    #                 smallest_mse_values_args.append(np.argmin(mse_values_test))
    #             m += 1
    #     # Find where the globally lowest architecture-level MSE is located
    #     overall_lowest_index = np.argmin(smallest_mse_values) #LLM tag: UiOGPT was used to identify the smallest mse value and its index, in this line and the two below
    #     # Use the SAME index in both arrays
    #     overall_lowest_mse = smallest_mse_values[overall_lowest_index]
    #     overall_lowest_mse_arg = smallest_mse_values_args[overall_lowest_index]
    #     print("\nSmallest MSE values for each tested architecture:")
    #     print(smallest_mse_values)
    #     print("\nIndices of the smallest MSE for each tested architecture:")
    #     print(smallest_mse_values_args)
    #     print("\nOverall lowest test MSE:")
    #     print("Lowest MSE:", overall_lowest_mse)
    #     print("Index/arg for this lowest MSE:", overall_lowest_mse_arg)

    #     # Optional: retrieve the actual learning rate rather than only its index
    #     print("Gamma value for lowest MSE:", gamma_values[overall_lowest_mse_arg])
    #     plt.hlines(overall_lowest_mse, 0, 0.5, label = "Lowest MSE", color = "#7c7c7c", linestyle = "dashed", alpha = 0.4)
    #     plt.axvline(x = gamma_values[overall_lowest_mse_arg], color = "#7c7c7c", label = f"$\\gamma$ for lowest MSE", linestyle = "dotted", alpha = 0.4)
    #     plt.xlabel(rf"Learning rate $\gamma$", fontsize=15)
    #     plt.ylabel("MSE", fontsize=15)
    #     plt.xticks(size=15)
    #     plt.yticks(size=15)
    #     plt.xscale("log")
    #     if met == "adam": 
    #         plt.xlim(8*10**-4, 0.2)
    #         plt.legend(fontsize=13, loc = "upper right")
    #     if met == "rmsprop":
    #         plt.xlim(7.5*10**-4, 5*10**-2)
    #         plt.legend(fontsize=13, loc = "lower right")
    #     plt.savefig(f"figures/oppgave_b_stochastic_{met}.pdf")
    #     plt.show()
    #     #------------------------------------------------------------------------------------------------------------------------------------------------


       






















# plt.plot(x_sorted, y_sorted,  "-", color = "#777977", label = "True function with noise", markersize = 3, alpha = 0.4)
# plt.plot(inputs_scaled, preds_100, "o", color = "#aa333c", label = "NN, two layers, size 100", markersize = 3)
# plt.plot(inputs_scaled, preds_50, "o", color = "#21534c", label = "NN, one layer, size 50", markersize = 3)
# # plt.plot(inputs_scaled, predictions_plain, "o", color = "#e5194f", label = "Neural network plain gradient decent layer size 100", markersize = 2)
# plt.plot(inputs_scaled, y_tilde, "o", color = "#ef9d2c", label = "Polyomial fit using OLS", markersize = 3)
# plt.plot(x_test, y_predict, "o", color = "#00ff6a", markersize = 3, label = "test 100")
# plt.ylim(-0.45, 0.9)
# plt.xticks(size = 13)
# plt.yticks(size = 13)
# plt.xlabel("x", fontsize = 14)
# plt.ylabel("y", fontsize = 14)
# plt.grid(alpha = 0.3)
# plt.legend(fontsize = 10)
# plt.savefig(f"figures/A/neural_network_start_{n}_points.pdf")
# plt.show()

# def predict_network(X, layers, activation_functions):
#     return feed_forward_batch(X, layers, activation_functions)

# plt.figure()



# plt.plot(x_sorted_no_noise, y_sorted_no_noise,  "-", color = "#777977", label = "True function no noise", markersize = 1, alpha = 0.7)
# print(len(inputs_scaled))
# print(len(preds_100))
# print(len(preds_50))
# print(len(y_tilde))
# print(len(x_sorted))
# print(len(y_sorted))
# plt.plot(x_sorted, y_sorted,  "-", color = "#777977", label = "True function with noise", markersize = 3, alpha = 0.4)
# plt.plot(inputs_scaled, preds_100, "o", color = "#aa333c", label = "NN, two layers, size 100", markersize = 3)
# plt.plot(inputs_scaled, preds_50, "o", color = "#21534c", label = "NN, one layer, size 50", markersize = 3)
# # plt.plot(inputs_scaled, predictions_plain, "o", color = "#e5194f", label = "Neural network plain gradient decent layer size 100", markersize = 2)
# plt.plot(inputs_scaled, y_tilde, "o", color = "#ef9d2c", label = "Polyomial fit using OLS", markersize = 3)
# plt.plot(x_test, y_predict, "o", color = "#00ff6a", markersize = 3, label = "test 100")
# plt.ylim(-0.45, 0.9)
# plt.xticks(size = 13)
# plt.yticks(size = 13)
# plt.xlabel("x", fontsize = 14)
# plt.ylabel("y", fontsize = 14)
# plt.grid(alpha = 0.3)
# plt.legend(fontsize = 10)
# plt.savefig(f"figures/A/neural_network_start_{n}_points.pdf")
# plt.show()

# def predict_network(X, layers, activation_functions):
#     return feed_forward_batch(X, layers, activation_functions)

# plt.figure()

