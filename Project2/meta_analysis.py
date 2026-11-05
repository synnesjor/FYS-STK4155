from source import utils_classification
from source import utils_regression

import numpy as np
import jax
from sklearn.model_selection import train_test_split
jax.config.update("jax_enable_x64", True)


# A meta analysis on the use of JAX automatic differentiation vs Backpropagation.

SEED = 2026
rng = np.random.default_rng(SEED) #LLM tag: ChatGPT suggested to make this more global, as we were having some issues with reproducibility

#-------------------------------------------------------- TEST CODE -----------------------------------------------------------------------------
sigma = 0.1
n = 2000

inputs = rng.uniform(-1,1,n).reshape(-1, 1)
inputs_scaled = utils_regression.inputs_scale(inputs)

targets = utils_regression.runge_1d(inputs_scaled, sigma, n)
targets_no_noise = utils_regression.runge_1d_no_noise(inputs_scaled)
targets_centered = targets - np.mean(targets)
targets_centered_no_noise = targets_no_noise - np.mean(targets_no_noise)

inputs_scaled, x_test, targets_centered, y_test = train_test_split(inputs_scaled, targets_centered, test_size = 0.2, random_state = 2026)
targets_test = utils_regression.runge_1d(x_test, sigma, len(x_test))

network_input_size = inputs_scaled.shape[1]
layer_output_sizes = [100] * 5 + [1]
activation_funcs = [utils_regression.leaky_ReLU] * 5 + [utils_regression.no_activation_func] 

layers = utils_regression.create_layers_batch(network_input_size, layer_output_sizes)
layers_JAX = utils_regression.gradient_func(inputs_scaled, layers, activation_funcs, targets_centered)

layers_backpropagation = utils_classification.backpropagation(inputs_scaled, targets_centered, layers, hidden_activation="lrelu", head="mse")

# finding difference between the two sets of layers
for i, (ad_grad, bp_grad) in enumerate(zip(layers_JAX, layers_backpropagation), start=1):
    # A deviation from theory section, implementation of JAX cost function does not
    # use half MSE, but the full, therefore all the subgradients are x2
    ad_W, ad_b = (np.array(value) * 0.5 for value in ad_grad)
    bp_W, bp_b = (np.array(value) for value in bp_grad) 


    print(f"\nLayer {i}")
    for parameter_name, ad_value, bp_value in (("W", ad_W, bp_W), ("b", ad_b, bp_b)):
        difference = ad_value - bp_value
        print(f" {parameter_name}: total error={np.sum(np.abs(difference)):.3e}, " f" shapes={ad_value.shape}")

