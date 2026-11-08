
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path

Path("data").mkdir(exist_ok=True)
from source.utils_regression import *


SEED = 2026
rng = np.random.default_rng(SEED) #LLM tag: ChatGPT suggested to make this more global, as we were having some issues with reproducibility

#-------------------------------------------------------- GENERAL PART -----------------------------------------------------------------------------
part = "part_d"
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

#-------------------------------------------------------- GENERAL PART -----------------------------------------------------------------------------




#-------------------------------------------------------- PART b) -----------------------------------------------------------------------------
if part == "part_b":
    """
    This part of the code does the initial analysis, where we calculate the MSE values for a range of learning rates. Standard architecture
    is 1 hidden layer of 50 nodes, or 2 hidden layers of 100 nodes, but these numbers are flexible. Standard sctivation function for hidden
    layers are sigmoid and the output layer has no activation function, but this may be changed also. When the optimal learning rate has been indentified, we 
    train our neural network, and compare it to the OLS results from Project1.

    General LLM tag: Claude Code Version 2.1.285 Opus 5.5 helped with the placement of plot commands and lists for 
    saving data (debugging). We verify by making sure the plots look like we want. UiOGPT was used to identify the minimum MSE values.
    """

    ols_deg11_fold25 = np.loadtxt("data/best_fit_from_project_1_x_and_y.txt")
    mse_train, mse_test_ols, theta, y_tilde, y_tilde_test, y_tilde_train, y_train, y_test= calculate_mse(inputs[:,0], targets[:,0], degree=7)

    colours = ["#aa333c", "#99d2fb", "#4e827b", "#ef9d2c", "#B0a6df", "#2da9cf", "#ff46a2", "#9aab64", "#e5194f", "#8f1cab", "#2da9cf", "#3ba336"]
    colours2 = ["#ef9d2c", "#020202", "#195626", "#ef2c2c", "#732775", "#a495cd", "#aaaf0f", "#f4b537", "#5513b9", "#5a585b", "#103a46", "#b83453"]
    colours3 = ["#f977a6", "#8e1c79" ]
    
    network_input_size = inputs_scaled.shape[1]


    # #for plotting the actual fit, PLAIN:
    # nodes = [50, 100]
    # n_hidden_layers = [1, 2]
            
    # m = 0 #just for colours in plot
    # plt.figure(figsize=(10, 9.5))
    
  
    # inputs_true = rng.uniform(-1,1,n).reshape(-1, 1)
    # inputs_true_scaled = inputs_scale(inputs)
    # targets_no_noise = runge_1d_no_noise(inputs_true_scaled)
    # targets_centered_no_noise = targets_no_noise-np.mean(targets_no_noise)

    # points = sorted(zip(inputs_true_scaled, targets_centered_no_noise)) # ChatGPT used for debugging this sorting
    # x_sorted = [p[0] for p in points]
    # y_sorted = [p[1] for p in points]

    # plt.plot(x_sorted, y_sorted, "--", color = "#1B1A1A", label = "Centered runge function")
    # plt.plot(inputs_scaled, targets_centered,  "o", color = "#4E5B4C", label = "Observed data", markersize = 3, alpha = 0.2)
    # plt.plot(x_test, y_tilde_test, "o", color = "#ef9d2c", label = "Polyomial fit using OLS, deg 6", markersize = 2)
    # plt.plot(ols_deg11_fold25[:,0], ols_deg11_fold25[:,1]+np.mean(targets), "o", color = "#729A72", label = "Polyomial fit using OLS, k-fold 25, deg 11", markersize = 2)

    # smallest_mse_values = []
    # smallest_mse_values_args = []

    # for idx, nod in enumerate(nodes):
    #     for n in n_hidden_layers:
    #         if (nod == 50 and n == 1) or (nod == 100 and n == 2):
    #             if (nod == 50 and n == 1):
    #                 gamma_value = 0.08446
    #             if (nod == 100 and n == 2):
    #                 gamma_value = 0.04481
    #             mse_values_train = []
    #             mse_values_test = []
    #             layer_output_sizes = [nod] * n + [1]
    #             activation_funcs = [sigmoid] * n + [no_activation_func] #make changes her if you wish
    #             layers_batch = create_layers_batch(network_input_size,layer_output_sizes)
    #             print("finding trained")
    #             trained_layers, history, preds, mse_v = train_network_gradient(inputs_scaled, layers_batch, activation_funcs, targets_centered, gamma = gamma_value, epochs=100)
    #             test_preds = predict_network(x_test,trained_layers,activation_funcs)
    #             mse_values_train.append(mse_v)
    #             test_mse = mse(test_preds, targets_test)
    #             mse_values_test.append(test_mse)
    #             # Plot the MSE values for this architecture
    #             # plt.plot(x_test, test_preds, "-o", markersize = 4, color = colours2[idx+m], label= f"{nod} nodes, {n} hidden layer{'s' if n > 1 else ''}, train")
    #             plt.plot(x_test, test_preds, "o", markersize = 2, color = colours[idx+m], label=f"{nod} nodes and {n} hidden layer{'s' if n > 1 else ''} NN, plain") #LLM tag: Claude helped with the placement of these plot commands

    #             print(f"MSE for {n} hidden layers of {nod} nodes:", mse_values_test)
    #         m += 1
    # print(mse_values_test)
    # plt.plot()
    # plt.xlabel("x", fontsize=24)
    # plt.ylabel("y", fontsize=24)
    # plt.xticks(size=23)
    # plt.yticks(size=23)
    # plt.legend(fontsize = 16)
    # plt.ylim(-0.38, 1.4)
    # plt.savefig(f"figures/b)/fit_to_runge_plain_ols_7_and_11.pdf")
    # plt.show()



    # #for plotting the actual fit, adam and rmsprop STOCHASTIC:
    # nodes = [50, 100]
    # n_hidden_layers = [1, 2]
    # degree_for_ols_fit = 7

    # methods = ["rmsprop"]

    # for met in methods:
    #     if met == "adam":
    #         gamma_value = 0.001 
    #         print("Doing adam!")
    #     if met == "rmsprop":
    #         gamma_value = 0.0045
    #         print("Doing rmsprop!")
            
    #     m = 0 #just for colours in plot
    #     plt.figure(figsize=(10, 9.5))
        
    #  
    #     inputs_true = rng.uniform(-1,1,n).reshape(-1, 1)
    #     inputs_true_scaled = inputs_scale(inputs)
    #     targets_no_noise = runge_1d_no_noise(inputs_true_scaled)
    #     targets_centered_no_noise = targets_no_noise-np.mean(targets_no_noise)
    
    #     points = sorted(zip(inputs_true_scaled, targets_centered_no_noise)) # ChatGPT used for debugging this sorting
    #     x_sorted = [p[0] for p in points]
    #     y_sorted = [p[1] for p in points]
    
    #     plt.plot(x_sorted, y_sorted, "--", color = "#1B1A1A", label = "Centered runge function")
    #     plt.plot(inputs_scaled, targets_centered,  "o", color = "#4E5B4C", label = "Observed data", markersize = 3, alpha = 0.2)
    #     plt.plot(x_test, y_tilde_test, "o", color = "#ef9d2c", label = f"Polyomial fit using OLS, deg {degree_for_ols_fit}", markersize = 2)
    #     plt.plot(ols_deg11_fold25[:,0], ols_deg11_fold25[:,1]+np.mean(targets), "o", color = "#729A72", label = "Polyomial fit using OLS, k-fold 25, deg 11", markersize = 2)

    #     smallest_mse_values = []
    #     smallest_mse_values_args = []

    #     for idx, nod in enumerate(nodes):
    #         for n in n_hidden_layers:
    #             if (nod == 50 and n == 1) or (nod == 100 and n == 2):
    #                 if (nod == 50 and n == 1) and met == "adam":
    #                     gamma_value = 0.007743
    #                 if (nod == 100 and n == 2) and met == "adam":
    #                     gamma_value = 0.001
    #                 if (nod == 50 and n == 1) and met == "rms_prop":
    #                     gamma_value = 0.004534
    #                 if (nod == 100 and n == 2) and met == "rms_prop":
    #                     gamma_value = 0.004534
    #                 mse_values_train = []
    #                 mse_values_test = []
    #                 layer_output_sizes = [nod] * n + [1]
    #                 activation_funcs = [sigmoid] * n + [no_activation_func]
    #                 layers_batch = create_layers_batch(network_input_size,layer_output_sizes)
    #                 print("finding trained")
    #                 trained_layers, history, preds, mse_v = train_network_stochastic_gradient(inputs_scaled, layers_batch, activation_funcs, targets_centered, gamma = gamma_value, method=met, schedule=None, epochs=100)
    #                 test_preds = predict_network(x_test,trained_layers,activation_funcs)
    #                 mse_values_train.append(mse_v)
    #                 test_mse = mse(test_preds, targets_test)
    #                 mse_values_test.append(test_mse)
    #                 # Plot the MSE values for this architecture
    #                 # plt.plot(x_test, test_preds, "-o", markersize = 4, color = colours2[idx+m], label= f"{nod} nodes, {n} hidden layer{'s' if n > 1 else ''}, train")
    #                 plt.plot(x_test, test_preds, "o", markersize = 2, color = colours[idx+m], label=f"{nod} nodes and {n} hidden layer{'s' if n > 1 else ''} NN, {met}") #LLM tag: Claude helped with the placement of these plot commands

    #                 print(f"MSE for {n} hidden layers of {nod} nodes:", mse_values_test)
    #             m += 1


    #     plt.plot()
    #     plt.xlabel("x", fontsize=24)
    #     plt.ylabel("y", fontsize=24)
    #     plt.xticks(size=23)
    #     plt.yticks(size=23)
    #     plt.legend(fontsize = 16)
    #     plt.ylim(-0.38, 1.4)
    #     plt.savefig(f"figures/b)/fit_to_runge_{met}_ols_7_and_11.pdf")
    #     plt.show()

    
    # For plotting the mse vs gamma value plots, and finding minumum mse and gamma value for PLAIN
    # gamma_values = np.logspace(-3,0,2)  
    gamma_values = np.logspace(-3, np.log10(0.3), 10)
    print("\nDoing PLAIN!\n")
    nodes = [50, 100]
    # nodes = [50]
    n_hidden_layers = [1, 2]

   
    plt.figure(figsize=(9.8, 7.7))
    smallest_mse_values = []
    smallest_mse_values_args = []

    for idx, nod in enumerate(nodes):
        for n in n_hidden_layers:
            if (nod == 50 and n == 1) or (nod == 100 and n == 2):
                mse_values_train = []
                mse_values_test = [] #LLM tag: Claude helped with the placement of the declariation of these lists
                layer_output_sizes = [nod] * n + [1]
                activation_funcs = [sigmoid] * n + [no_activation_func]
                for gam in gamma_values:
                    print(f"for gamma = {gam}")
                    layers_batch = create_layers_batch(network_input_size, layer_output_sizes)
                    trained_layers, history, preds, mse_v = train_network_gradient(inputs_scaled, layers_batch, activation_funcs, targets_centered, gamma=gam)
                    # test_preds = predict_network(x_test, trained_layers, activation_funcs)
                    mse_values_train.append(mse_v)
                    # test_mse = mse(test_preds, targets_test)
                    # mse_values_test.append(test_mse)
                plt.plot(gamma_values, mse_values_train, "-o", color = colours3[idx], label= f" train: {nod} nodes, {n} hidden layer{'s' if n > 1 else ''}") #LLM tag: Claude helped with the placement of these plot commands
                # print("mse test is", mse_values_test)
                smallest_mse_values.append(np.min(mse_values_train))
                smallest_mse_values_args.append(np.argmin(mse_values_train))
        # print(np.min(mse_values_train), np.argmin(mse_values_train))
        # print(np.min(mse_values_test), np.argmin(mse_values_test))

            overall_lowest_index = np.argmin(smallest_mse_values) #LLM tag: UiOGPT was used to identify the smallest mse value and its index, in this line and the two below
            overall_lowest_mse = smallest_mse_values[overall_lowest_index]
            overall_lowest_mse_arg = smallest_mse_values_args[overall_lowest_index]

            if (nod == 50 and n == 1) or (nod == 100 and n == 2):
                print(f"\nFor {n} hidden layers and {nod} nodes:")
                print("Overall lowest test MSE:")
                print("Lowest MSE:", overall_lowest_mse)
                print("Index/arg for this lowest MSE:", overall_lowest_mse_arg)


        print("Gamma value for lowest MSE:", gamma_values[overall_lowest_mse_arg], "\n")
        plt.hlines(overall_lowest_mse, 0, 0.5, color = "#7c7c7c", linestyle = "dashed", alpha = 0.4)
        plt.axvline(x = gamma_values[overall_lowest_mse_arg], color = "#7c7c7c", linestyle = "dotted", alpha = 0.4)
    plt.xlabel(rf"Learning rate $\gamma$", fontsize = 20)
    plt.ylabel("MSE", fontsize = 20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xscale("log")
    plt.ylim(0.0, 0.1)
    plt.legend(fontsize = 16)
    # plt.savefig(f"figures/b)/oppgave_b_plain_mse_vs_gamma.pdf")
    plt.show()


    # For plotting the mse vs gamma value plots, and finding minumum mse and gamma value for STOCHASTIC
    # nodes = [50, 100]
    nodes = [50,100]
    n_hidden_layers = [1, 2]

    methods = ["adam"]

    # print("\nDoing STOCHASIC {method}!\n")

    for met in methods:
        if met == "adam":
            gamma_values = np.logspace(-3, -1, 10)
            print("\nDoing STOCHASIC adam!\n")
        if met == "rmsprop":
            gamma_values = np.logspace(-3, np.log10(0.03), 10)
            print("\nDoing STOCHASIC rmsprop!\n")
        m = 0 #just for colours in plot
        if met == "adam":
            plt.figure(figsize=(9.8, 7.7))
        if met == "rmsprop":
            plt.figure(figsize=(9.8, 7.7))

        smallest_mse_values = []
        smallest_mse_values_args = []
    
        for idx, nod in enumerate(nodes):
            for n in n_hidden_layers:
                if (nod == 50 and n == 1) or (nod == 100 and n == 2):
                    mse_values_train = []
                    mse_values_test = []
                    layer_output_sizes = [nod] * n + [1]
                    activation_funcs = [sigmoid] * n + [no_activation_func]
                    for gam in gamma_values:
                        print(f"for gamma = {gam}")
                        layers_batch = create_layers_batch(network_input_size,layer_output_sizes)
                        trained_layers, history, preds, mse_v = train_network_stochastic_gradient(inputs_scaled, layers_batch, activation_funcs, targets_centered, gamma=gam, method=met, schedule=None, epochs=100)
                        # test_preds = predict_network(x_test,trained_layers,activation_funcs)
                        mse_values_train.append(mse_v)
                        # test_mse = mse(test_preds, targets_test)
                        # mse_values_test.append(test_mse)
                    # Plot the MSE values for this architecture
                    plt.plot(gamma_values, mse_values_train, "-o", markersize = 4, color = colours3[idx], label= f"{nod} nodes, {n} hidden layer{'s' if n > 1 else ''}, train")
                    # plt.plot(gamma_values, mse_values_test, "-o", markersize = 4, color = colours[idx+m], label=f"{nod} nodes, {n} hidden layer{'s' if n > 1 else ''}, test") #LLM tag: Claude helped with the placement of these plot commands
                    # Save the lowest MSE for THIS architecture,
                    # plus the index where it occurs in mse_values_test.
                    smallest_mse_values.append(np.min(mse_values_train))
                    smallest_mse_values_args.append(np.argmin(mse_values_train))
                m += 1
        # Find where the globally lowest architecture-level MSE is located
                overall_lowest_index = np.argmin(smallest_mse_values) #LLM tag: UiOGPT was used to identify the smallest mse value and its index, in this line and the two below
                # Use the SAME index in both arrays
                overall_lowest_mse = smallest_mse_values[overall_lowest_index]
                overall_lowest_mse_arg = smallest_mse_values_args[overall_lowest_index]

                if (nod == 50 and n == 1) or (nod == 100 and n == 2):
                    print(f"\nFor {n} hidden layers and {nod} nodes:")
                    print("Overall lowest test MSE:")
                    print("Lowest MSE:", overall_lowest_mse)
                    print("Index/arg for this lowest MSE:", overall_lowest_mse_arg)


        # Optional: retrieve the actual learning rate rather than only its index
            print("Gamma value for lowest MSE:", gamma_values[overall_lowest_mse_arg], "\n")
            plt.hlines(overall_lowest_mse, 0, 0.5, color = "#7c7c7c", linestyle = "dashed", alpha = 0.4)
            plt.axvline(x = gamma_values[overall_lowest_mse_arg], color = "#7c7c7c", linestyle = "dotted", alpha = 0.4)
        plt.xlabel(rf"Learning rate $\gamma$", fontsize=20)
        plt.ylabel("MSE", fontsize=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.xscale("log")
        if met == "adam": 
            plt.xlim(8*10**-4, 0.2)
            plt.ylim(0.009, 0.014)
            plt.legend(fontsize=16, loc = "upper right")
        if met == "rmsprop":
            plt.xlim(7.5*10**-4, 5*10**-2)
            plt.ylim(0.007, 0.025)
            plt.legend(fontsize=16, loc = "upper left")
        plt.savefig(f"figures/b)/oppgave_b_stochastic_{met}_mse_vs_gamma.pdf")
        plt.show()
#-------------------------------------------------------- PART b) -----------------------------------------------------------------------------


#-------------------------------------------------------- PART c) -----------------------------------------------------------------------------
if part == "part_c":
    """
    This part investigates the number of hidden layers and nodes; our goal is to indentify the optimal combination of these hyperparameters. We also 
    test various activation functions and optimisers for the gradient decent.
    Running this full part TAKES TIME! That is why we run this and save the data and then plot, we do not olot in the run. The plotting script
    can be found in the plotting_scripts folder. 
    
    General LLM tag: ChatGPT Version GPT-5.6 Luna Reasoning was used in order to write the code so that we would get the desired text files. 
    We check that the outcoming values are the same as if we were to print it out manually and more split up, and we confirm that we get the same. 
    """
    
    mse_train, mse_test_ols, theta, y_tilde, y_tilde_test, y_tilde_train, y_train, y_test= calculate_mse(inputs[:,0], targets[:,0], degree=7)
    act_functions = [leaky_ReLU]
    # act_functions = [sigmoid]
    act_functions_names = ["sigmoid", "leaky_ReLU", "ReLU"]
    # act_functions_names = ["sigmoid"]
    n_hidden_layers = [1, 2, 3, 5, 10]
    number_of_nodes = [2, 5, 10, 20, 50, 100, 200]
    gamma_value = 0.001
    met = "adam"
    # n_hidden_layers = [1] 

    colours = ["#aa333c", "#99d2fb", "#4e827b", "#ef9d2c", "#B0a6df", "#2da9cf", "#ff46a2", "#9aab64", "#e5194f", "#8f1cab", "#2da9cf", "#3ba336"]
    colours2 = ["#ef9d2c", "#020202", "#195626", "#ef2c2c", "#732775", "#a495cd", "#aaaf0f", "#f4b537", "#5513b9", "#5a585b", "#103a46", "#b83453"]

    
    network_input_size = inputs_scaled.shape[1]

    """
    LLM tag here: ChatGPT was used in order to write the code so that we would get the desired text files. We check that the outcoming
    values are the same as if we were to print it out manually and more split up, and we confire that we get the same. 
    """
    m = 0
    #for c)
    plt.figure()
    Path("data").mkdir(exist_ok=True)
    for idx, a in enumerate(act_functions):
        mse_values_train = []
        mse_values_test = []
        for n in n_hidden_layers:
            mse_for_n_layers = []
            for nod in number_of_nodes:
                layer_output_sizes = [nod] * n + [1]
                activation_funcs = [a] * n + [no_activation_func] #LLM use: Claude used to efficiently create the array for activation functions
                print(activation_funcs)
                layers_batch = create_layers_batch(network_input_size, layer_output_sizes)
                predictions = feed_forward_batch(inputs_scaled, layers_batch, activation_funcs)
                layers_grad = gradient_func(inputs_scaled, layers_batch, activation_funcs, targets_centered)
                trained_layers, history, preds, mse_v = train_network_stochastic_gradient(inputs_scaled, layers_batch, activation_funcs, targets_centered, gamma = gamma_value, method = met, schedule=None, epochs=100)
                mse_values_train.append(mse_v)
                for_test_data = predict_network(x_test, trained_layers, activation_funcs)
                mse_for_test_data = mse(for_test_data, targets_test)
                mse_for_n_layers.append(mse_for_test_data)
            mse_values_test.append(mse_for_n_layers)
        mse_values_test = np.array(mse_values_test)

        filename = f"data/mse_{act_functions_names[idx]}_{met}.txt"

        # Exactly 5 rows, each with 7 MSE values
        np.savetxt(filename, mse_values_test, fmt="%.10e", delimiter=" ")
                
        # plt.plot(n_hidden_layers, mse_values_train, "-o", color = colours[m], label = f"{act_functions_names[idx]} train")
        plt.plot(n_hidden_layers, mse_values_test, "-o", color = colours[m+4], label = f"{act_functions_names[idx]} test")
        plt.legend()
        m+=1
    # plt.savefig("figures/c)/exercise_c_mse_vs_hidden_layers_stochastic_{met}_new.pdf")
    plt.show()
#-------------------------------------------------------- PART c) -----------------------------------------------------------------------------
    
# Cross validation of learning rate

from sklearn.model_selection import KFold

def cross_validate_network(X, y, layer_output_sizes, activation_funcs, method, gamma=0.01, norm="l2", lam=0.0, epochs=100, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    validation_mse = []
    train_mse_values = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):

        layers = create_layers_batch(X.shape[1], layer_output_sizes)

        if method == "plain":
            print("using standard gradient")
            trained_layers, _, _, train_mse = train_network_gradient(
                    X[train_idx],
                    layers,
                    activation_funcs,
                    y[train_idx],
                    gamma=gamma,
                    norm=norm,
                    epochs=epochs,
                    method=method,
                    lam=lam
                )

        else:
            print("using stochasic:")
            trained_layers, _, _, train_mse = train_network_stochastic_gradient(
                    X[train_idx],
                    layers,
                    activation_funcs,
                    y[train_idx], 
                    schedule = None, 
                    gamma=gamma,
                    norm = norm,
                    epochs=epochs,
                    method=method,
                )

        train_mse_values.append(train_mse)

        val_predictions = predict_network(X[val_idx], trained_layers, activation_funcs)
        fold_mse = float(mse(val_predictions,y[val_idx]))

        validation_mse.append(fold_mse)
        # print(f"Fold {fold + 1}: MSE = {fold_mse:.6g}")

    validation_mse = np.array(validation_mse)

    return np.mean(validation_mse), np.std(validation_mse, ddof = 1) / np.sqrt(n_splits)

#-------------------------------------------------------- PART f) -----------------------------------------------------------------------------
if part == "part_f":
    """
    In this part we perform a k-fold cross-validation for our regression problem, investigating the learning rate and the penalty parameter.
    Running this full part TAKES TIME! That is why we run this and save the data and then plot, we do not olot in the run. The plotting script
    can be found in the plotting_scripts folder.   
    """

    # method = ["plain", "adam", "rmsprop"]
    
    method = ["plain"]
    for met in method:
        print(f"using {met}:")
        layer_output_sizes = [100] * 5 + [1]
        activation_funcs = [leaky_ReLU] * 5 + [no_activation_func] #LLM use: Claude used to efficiently create the array for activation functions

        if met == "plain":
            # gamma_values = np.logspace(-3, 0, 10)
            gamma_values = [10**-3, 10**-2, 5*10**-2, 10**-1, 5*10**-1]
            # lambda_values = np.concatenate(([0.0], np.logspace(-5, -3, 5)))
            lambda_values = [0, 10**-5, 10**-4, 5*10**-4, 10**-3]
            # lambda_values = np.logspace(-5, -3, 10)
        
        if met == "adam":
            gamma_values = [10**-3, 10**-2, 5*10**-2, 10**-1, 5*10**-1]
            lambda_values = [0, 10**-5, 10**-4, 5*10**-4, 10**-3]

        if met == "rmsprop":
            gamma_values = [10**-3, 10**-2, 5*10**-2, 10**-1, 5*10**-1]
            lambda_values = [0, 10**-5, 10**-4, 5*10**-4, 10**-3]

        best_parameter = None
        best_mse = 1e6 # Choose high MSE

        validation_mse_values = []
        validation_save_parameters = []

        validation_mse_grid = np.zeros((len(gamma_values), len(lambda_values)))
        # Path("data").mkdir(exist_ok=True)
        # filename = f"data/mse_crossvalidation_test.txt"
        for gamma_idx, gamma in enumerate(gamma_values):
            for lam_idx, lam in enumerate(lambda_values):
                cv_mean, cv_sd = cross_validate_network(inputs_scaled, targets_centered, layer_output_sizes, activation_funcs, method = met, gamma=gamma, norm = "l1", lam=lam, epochs=100, n_splits=3)
                # print(f"MSE Mean {cv_mean} SD {cv_sd}")
                
                validation_mse_values.append(cv_mean)
                validation_save_parameters.append((gamma, lam))

                if np.isnan(cv_mean) or cv_mean > 0.1:
                    print("yes nan")
                    validation_mse_grid[gamma_idx, lam_idx] = 0.1
                else:
                    validation_mse_grid[gamma_idx, lam_idx] = cv_mean
                    

                if cv_mean < best_mse:
                    best_parameter = (gamma, lam)
                    best_mse = cv_mean
                    threshold = cv_mean + cv_sd
                    print(f"New gamma: {gamma}")

        filename = f"data/mse_crossvalidation_{met}_l1_only_five.txt"

        np.savetxt(filename, validation_mse_grid, fmt="%.10e", delimiter=" ")

        mse_values = np.array(validation_mse_values)
        within_one_sd = mse_values <= threshold
        eligible_indices = np.where(within_one_sd)[0]

        print(best_parameter)
        
        for i in eligible_indices:
            print(f"Eligible parameters: {validation_save_parameters[i]} " f"with MSE {mse_values[i]:.4f} ")

        print("Best learning rate gamma from cross validation for method plain", best_parameter[0])
        print("Best penalty parameter lamda from cross validation for method plain", best_parameter[1])

if part == "part_d":
    layer_sizes = [5, 1]
    activation_functions = [leaky_ReLU,leaky_ReLU, leaky_ReLU, leaky_ReLU, no_activation_func]
    layers = create_layers_batch(inputs_scaled.shape[1], layer_sizes)

    lambda_values = np.concatenate(([0.0], np.logspace(-5, -3, 10)))

    mse_train_l1, mse_train_l2 = [],[]

    #colours = ["#aa333c", "#99d2fb", "#4e827b", "#ef9d2c", "#B0a6df", "#2da9cf", "#ff46a2", "#9aab64", "#e5194f", "#8f1cab", "#2da9cf", "#3ba336"]
    colours2 = ["#ef9d2c", "#020202", "#195626", "#ef2c2c", "#732775", "#a495cd", "#aaaf0f", "#f4b537", "#5513b9", "#5a585b", "#103a46", "#b83453"]

    plt.figure()
    for lambda_val in lambda_values:
        # runs for l2 - norm
        layers_l2, history_l2, predictions_l2, mse_value_l2 = train_network_stochastic_gradient(
            inputs_scaled,
            layers,
            activation_functions,
            targets_centered,
            gamma=0.001,
            epochs=100,
            norm="l2", # change here to change norm, l1 or l2
            lam=lambda_val,
            schedule=None,
            method="adam"
        )
        #test_predictions_l2 = predict_network(x_test, layers_l2, activation_functions)
        #mse_train_l2 = mse(predictions_l2, targets_centered)
        mse_train_l2.append(mse(predictions_l2, targets_centered))
        #print(f"train MSE={float(mse_train):.6f}, test MSE={float(mse_test):.6f}")

        # runs for l1 - norm
        layers_l1, history_l1, predictions_l1, mse_value_l1 = train_network_stochastic_gradient(
            inputs_scaled,
            layers,
            activation_functions,
            targets_centered,
            gamma=0.001,
            epochs=100,
            norm="l1", # change here to change norm, l1 or l2
            lam=lambda_val,
            schedule=None,
            method="adam"
        )
        #test_predictions_l1 = predict_network(x_test, layers_l1, activation_functions)
        #mse_train = mse(predictions, targets_centered)
        mse_train_l1.append(mse(predictions_l1, targets_centered))
        #print(f"train MSE={float(mse_train):.6f}, test MSE={float(mse_test):.6f}")

    plt.plot(lambda_values, mse_train_l1, "-o", color = "#ef9d2c", label = r"$\ell_1$-norm", markersize = 1, alpha = 0.5)
    plt.plot(lambda_values, mse_train_l2, "-o", color = "#732775", label = r"$\ell_2$-norm", markersize = 1, alpha = 0.5)
    plt.xscale("symlog", linthresh=1e-5)
    plt.xlabel(r"$\lambda$", fontsize=15) 
    plt.ylabel(r"MSE($\lambda$)", fontsize=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.legend(fontsize = 15)
    plt.tight_layout()
    plt.savefig("figures/d)/MSE_l1_l2_norm.pdf", bbox_inches="tight")
    #plt.show()       
      
if part == "plot_f":
    """
    This part plots the final fits for our neural network, and compares it to the Ridge analysis results from Project1, and 
    the neural network baseline(when using rms prop).

    General LLM tag: Claude Code Version 2.1.285 Opus 5.5 helped with the placement of plot commands (debugging). We verify by making sure the plots look like we want.
    """
    
    
    act_functions = [sigmoid, leaky_ReLU, ReLU]
    act_functions_names = ["sigmoid", "leaky_ReLU", "ReLU"]
    n_hidden_layers = [1,2,3,5]
    # n_hidden_layers = [1] 

    colours = ["#aa333c", "#99d2fb", "#4e827b", "#ef9d2c", "#B0a6df", "#2da9cf", "#ff46a2", "#9aab64", "#e5194f", "#8f1cab", "#2da9cf", "#3ba336"]
    colours2 = ["#ef9d2c", "#020202", "#195626", "#ef2c2c", "#732775", "#a495cd", "#aaaf0f", "#f4b537", "#5513b9", "#5a585b", "#103a46", "#b83453"]
    
    network_input_size = inputs_scaled.shape[1]

    baseline_txtfile = np.loadtxt("data/base_line_test_preds.txt")
    baseline_vals = baseline_txtfile[:]
    ridge_comparison_txtfile = np.loadtxt("data/ridge_data_from_project1_deg_12_1.4174741629268048e-05_5.txt")
    ridge_vals_x = ridge_comparison_txtfile[:,0]
    ridge_vals_y = ridge_comparison_txtfile[:,1]

    inputs_true = rng.uniform(-1,1,n).reshape(-1, 1)
    inputs_true_scaled = inputs_scale(inputs)
    targets_no_noise = runge_1d_no_noise(inputs_true_scaled)
    targets_centered_no_noise = targets_no_noise-np.mean(targets_no_noise)

    points = sorted(zip(inputs_true_scaled, targets_centered_no_noise))# ChatGPT used for debugging this sorting
    x_sorted = [p[0] for p in points]
    y_sorted = [p[1] for p in points]

    methods_list = ["plain", "adam", "rmsprop"]
    # methods_list = ["rmsprop"]
    
    for method in methods_list:
        if method == "plain":
            print("running plain")
            layer_output_sizes = [100] * 5 + [1]
            activation_funcs = [leaky_ReLU] * 5 + [no_activation_func] #LLM use: Claude used to efficiently create the array for activation functions
            gamma_value = 0.1
            lambda_value = 0.0005
            norm = "l2"
            method = "plain"


            m = 0 #just for colours in plot
            plt.figure(figsize=(10, 9.5))


            # plt.plot(inputs_scale(np.linspace(np.min(inputs_scaled), np.min(inputs_scaled), 5000)), runge_1d_no_noise(np.linspace(np.min(inputs_scaled), np.min(inputs_scaled), 5000)) + np.mean(runge_1d_no_noise(np.linspace(np.min(inputs_scaled), np.min(inputs_scaled), 5000))))
            plt.plot(x_sorted, y_sorted, "--", color = "#1B1A1A", label = "Centered runge function")
            plt.plot(inputs_scaled, targets_centered,  "o", color = "#4E5B4C", label = "Observed data", markersize = 3, alpha = 0.2)
            # plt.plot(x_test, y_tilde_test, "o", color = "#ef9d2c", label = "Polyomial fit using OLS, deg 6", markersize = 2)
            # plt.plot(ols_deg11_fold25[:,0], ols_deg11_fold25[:,1]+np.mean(targets), "o", color = "#729A72", label = "Polyomial fit using OLS, k-fold 25, deg 11", markersize = 2)
            plt.plot(ridge_vals_x, ridge_vals_y + np.mean(targets), "o", color = "#198bb1", label = "Polynomial fit using Ridge, 5 k-folds, deg 12 ", markersize = 2)


            layers_batch = create_layers_batch(network_input_size,layer_output_sizes)
            print("finding trained")
            trained_layers_after_crossval, history_after_cross_val, preds_after_cross_val, mse_v_after_cross_val = train_network_gradient(inputs_scaled, layers_batch, activation_funcs, targets_centered, gamma = gamma_value, epochs=100, norm = norm, lam = lambda_value)


            test_preds = predict_network(x_test,trained_layers_after_crossval,activation_funcs)
            # mse_values_train.append(mse_v)
            test_mse = mse(test_preds, targets_test)
            print(test_mse)
            # mse_values_test.append(test_mse)
            # Plot the MSE values for this architecture
            # plt.plot(x_test, test_preds, "-o", markersize = 4, color = colours2[idx+m], label= f"{nod} nodes, {n} hidden layer{'s' if n > 1 else ''}, train")
            plt.plot(x_test, test_preds, "o", markersize = 3, color = "#cf2b7f", label = "Cross validated fit for NN, plain")

            # print(f"MSE for {n} hidden layers of {nod} nodes:", mse_values_test)
                    
            # print(mse_values_test)
            plt.plot()
            plt.xlabel("x", fontsize=24)
            plt.ylabel("y", fontsize=24)
            plt.xticks(size=23)
            plt.yticks(size=23)
            plt.legend(fontsize = 16)
            plt.ylim(-0.37, 1.3)
            plt.savefig(f"figures/f)/final_fit_plain_l2_compare.pdf")
            plt.show()

        if method == "adam": 
            print("running adam:")
            layer_output_sizes = [100] * 5 + [1]
            activation_funcs = [leaky_ReLU] * 5 + [no_activation_func] #LLM use: Claude used to efficiently create the array for activation functions
            gamma_value = 0.001
            lambda_value = 0.001
            norm = "l2"
            method = "adam"


            m = 0 #just for colours in plot
            plt.figure(figsize=(10, 9.5))

            """
            LLM tag:
            The three lines below, used for sorting the points for the true function, was written by ChatGPT.
            """

            plt.plot(x_sorted, y_sorted, "--", color = "#1B1A1A", label = "Centered runge function")
            plt.plot(inputs_scaled, targets_centered,  "o", color = "#4E5B4C", label = "Observed data", markersize = 3, alpha = 0.2)
            # plt.plot(x_test, y_tilde_test, "o", color = "#ef9d2c", label = "Polyomial fit using OLS, deg 6", markersize = 2)
            # plt.plot(ols_deg11_fold25[:,0], ols_deg11_fold25[:,1]+np.mean(targets), "o", color = "#729A72", label = "Polyomial fit using OLS, k-fold 25, deg 11", markersize = 2)
            plt.plot(ridge_vals_x, ridge_vals_y + np.mean(targets), "o", color = "#198bb1", label = "Polynomial fit using Ridge, 5 k-folds, deg 12 ", markersize = 2)

            # layer_output_sizes = [100] * 5 + [1]
            # activation_funcs = [leaky_ReLU] * 5 + [no_activation_func]
            layers_batch = create_layers_batch(network_input_size,layer_output_sizes)
            print("finding trained")
            trained_layers_after_crossval, history_after_cross_val, preds_after_cross_val, mse_v_after_cross_val = train_network_stochastic_gradient(inputs_scaled, layers_batch, activation_funcs, targets_centered, method, gamma = gamma_value, epochs=100, norm = norm, lam = lambda_value)


            test_preds = predict_network(x_test,trained_layers_after_crossval,activation_funcs)
            # mse_values_train.append(mse_v)
            test_mse = mse(test_preds, targets_test)
            print(test_mse)
            # Plot the MSE values for this architecture
            # plt.plot(x_test, test_preds, "-o", markersize = 4, color = colours2[idx+m], label= f"{nod} nodes, {n} hidden layer{'s' if n > 1 else ''}, train")
            plt.plot(x_test, test_preds, "o", markersize = 3, color = "#cf2b7f", label = "Cross validated fit for NN, adam") #LLM tag: Claude helped with the placement of these plot commands
        
            # print(f"MSE for {n} hidden layers of {nod} nodes:", mse_values_test)
                    
            # print(mse_values_test)
            plt.plot()
            plt.xlabel("x", fontsize=24)
            plt.ylabel("y", fontsize=24)
            plt.xticks(size=23)
            plt.yticks(size=23)
            plt.legend(fontsize = 16)
            plt.ylim(-0.35, 1.25)
            plt.savefig(f"figures/f)/final_fit_adam_l2_compare.pdf")
            plt.show()

        if method == "rmsprop": 
            print("running rmsprop:")
               

            layer_output_sizes = [100] * 5 + [1]
            activation_funcs = [leaky_ReLU] * 5 + [no_activation_func] #LLM use: Claude used to efficiently create the array for activation functions
            gamma_value = 0.001
            lambda_value = 0.0005
            norm = "l2"
            method = "rmsprop"


            m = 0 #just for colours in plot
            plt.figure(figsize=(10, 9.5))

            """
            LLM tag:
            The three lines below, used for sorting the points for the true function, was written by ChatGPT.
            """
            
            plt.plot(x_sorted, y_sorted, "--", color = "#1B1A1A", label = "Centered runge function")
            plt.plot(inputs_scaled, targets_centered,  "o", color = "#4E5B4C", label = "Observed data", markersize = 3, alpha = 0.2)
            # plt.plot(x_test, y_tilde_test, "o", color = "#ef9d2c", label = "Polyomial fit using OLS, deg 6", markersize = 2)
            # plt.plot(ols_deg11_fold25[:,0], ols_deg11_fold25[:,1]+np.mean(targets), "o", color = "#729A72", label = f"Polyomial fit using OLS, 25 k-folds, deg 11", markersize = 2)
            plt.plot(ridge_vals_x, ridge_vals_y + np.mean(targets), "o", color = "#198bb1", label = "Polynomial fit using Ridge, 5 k-folds, deg 12 ", markersize = 2)

            # layer_output_sizes = [100] * 5 + [1]
            # activation_funcs = [leaky_ReLU] * 5 + [no_activation_func]
            layers_batch = create_layers_batch(network_input_size,layer_output_sizes)
            print("finding trained")
            trained_layers_after_crossval, history_after_cross_val, preds_after_cross_val, mse_v_after_cross_val = train_network_stochastic_gradient(inputs_scaled, layers_batch, activation_funcs, targets_centered, method, gamma = gamma_value, epochs=100, norm = norm, lam = lambda_value)


            test_preds = predict_network(x_test,trained_layers_after_crossval,activation_funcs)
            # mse_values_train.append(mse_v)
            test_mse = mse(test_preds, targets_test)
            print(test_mse)
            # Plot the MSE values for this architecture
            # plt.plot(x_test, test_preds, "-o", markersize = 4, color = colours2[idx+m], label= f"{nod} nodes, {n} hidden layer{'s' if n > 1 else ''}, train")
            plt.plot(x_test, test_preds, "o", markersize = 3, color = "#cf2b7f", label = "Cross validated fit for NN, rmsprop") #LLM tag: Claude helped with the placement of these plot commands
            plt.plot(x_test, baseline_vals, "o", markersize = 2, color = "#882bcf", label = "Baseline NN")

            # print(f"MSE for {n} hidden layers of {nod} nodes:", mse_values_test)
                    
            # print(mse_values_test)
            plt.plot()
            plt.xlabel("x", fontsize=24)
            plt.ylabel("y", fontsize=24)
            plt.xticks(size=23)
            plt.yticks(size=23)
            plt.legend(fontsize = 16)
            plt.ylim(-0.35, 1.25)
            plt.savefig(f"figures/f)/final_fit_rmsprop_l2_compare.pdf")
            plt.show()


if part == "baseline":
    """
    This part calculates the baseline for our regnression neural network. This is done by removing all hidden layers, and not including an activation function 
    in the output layer. We save the predicted baseline values to a textfile, and use them in the plotting script below in part plot_f. 
    """ 
    

    ols_deg11_fold25 = np.loadtxt("data/best_fit_from_project_1_x_and_y.txt")
    mse_train, mse_test_ols, theta, y_tilde, y_tilde_test, y_tilde_train, y_train, y_test= calculate_mse(inputs[:,0], targets[:,0], degree=7)
    act_functions = [sigmoid, leaky_ReLU, ReLU]
    act_functions_names = ["sigmoid", "leaky_ReLU", "ReLU"]
    n_hidden_layers = [1,2,3,5]
    # n_hidden_layers = [1] 

    colours = ["#aa333c", "#99d2fb", "#4e827b", "#ef9d2c", "#B0a6df", "#2da9cf", "#ff46a2", "#9aab64", "#e5194f", "#8f1cab", "#2da9cf", "#3ba336"]
    colours2 = ["#ef9d2c", "#020202", "#195626", "#ef2c2c", "#732775", "#a495cd", "#aaaf0f", "#f4b537", "#5513b9", "#5a585b", "#103a46", "#b83453"]
    
    network_input_size = inputs_scaled.shape[1]    

    print("running rmsprop:")
    layer_output_sizes = [1]
    activation_funcs = [no_activation_func] #LLM use: Claude used to efficiently create the array for activation functions
    gamma_value = 0.001
    lambda_value = 0.0005
    norm = "l2"
    method = "rmsprop"


    m = 0 #just for colours in plot
    plt.figure(figsize=(9.0, 9.5))

    points = sorted(zip(inputs_scaled, targets_centered))# ChatGPT used for debugging this sorting
    x_sorted = [p[0] for p in points]
    y_sorted = [p[1] for p in points]
    plt.plot(x_sorted, y_sorted,  "-", color = "#B5B4B4", label = "True function", markersize = 1, alpha = 0.5)
    # plt.plot(x_test, y_tilde_test, "o", color = "#ef9d2c", label = "Polyomial fit using OLS, deg 6", markersize = 2)
    # plt.plot(ols_deg11_fold25[:,0], ols_deg11_fold25[:,1]+np.mean(targets), "o", color = "#729A72", label = "Polyomial fit using OLS, k-fold 25, deg 11", markersize = 2)


    # layer_output_sizes = [100] * 5 + [1]
    # activation_funcs = [leaky_ReLU] * 5 + [no_activation_func]
    layers_batch = create_layers_batch(network_input_size,layer_output_sizes)
    print("finding trained")
    trained_layers_after_crossval, history_after_cross_val, preds_after_cross_val, mse_v_after_cross_val = train_network_stochastic_gradient(inputs_scaled, layers_batch, activation_funcs, targets_centered, method, gamma = gamma_value, epochs=100, norm = norm, lam = lambda_value)




    baseline_preds = predict_network(x_test,trained_layers_after_crossval,activation_funcs)
    test_preds = baseline_preds
    # mse_values_train.append(mse_v)
    test_mse = mse(test_preds, targets_test)
    print("baseline mse:", test_mse)
    # Plot the MSE values for this architecture
    # plt.plot(x_test, test_preds, "-o", markersize = 4, color = colours2[idx+m], label= f"{nod} nodes, {n} hidden layer{'s' if n > 1 else ''}, train")
    plt.plot(x_test, test_preds, "o", markersize = 2, color = "#cf2b7f", label = "Cross validated fit")
    # for each in test_preds:
    #     print(each[0])
    print("mean of test_preds is:", np.mean(test_preds))

    # print(f"MSE for {n} hidden layers of {nod} nodes:", mse_values_test)
            
    # print(mse_values_test)
    plt.plot()
    plt.xlabel("x", fontsize=21)
    plt.ylabel("y", fontsize=21)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.legend(fontsize = 16)
    # plt.ylim(-0.33, 1.4)
    plt.savefig(f"figures/f)/final_fit_rmsprop_l2_baseline.pdf")
    plt.show()
#-------------------------------------------------------- PART f) -----------------------------------------------------------------------------