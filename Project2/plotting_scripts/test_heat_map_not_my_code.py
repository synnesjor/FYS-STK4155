import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


"""
LLM tag: ChatGPT wrote the sceleton of this short plotting code; we have changed it to work exactly how we wanted it to. 
The code snippet for modifying the colour map was written by ChatGPT and checked by us to make sure it worked as we wanted to.  
"""

# Original colormap
cmap = plt.cm.RdPu

# Remove the lightest 15% of the color scale
new_colors = cmap(np.linspace(0.20, 1, 256))

# Create a new colormap
new_cmap = mcolors.LinearSegmentedColormap.from_list(
    'RdPu_darker', new_colors
)


method = ["leaky_ReLU", "sigmoid", "ReLU"]

for each in method:

    mse = np.loadtxt(f"textfiles/mse_{each}_adam_train.txt")

    n_hidden_layers = [1, 2, 3, 5, 10]
    number_of_nodes = [2, 5, 10, 20, 50, 100, 200]

    ax = sns.heatmap(
        mse, annot=True, annot_kws={"size": 12}, fmt=".3f", cmap = new_cmap, xticklabels=number_of_nodes, yticklabels=n_hidden_layers)


    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    plt.xlabel("Nodes per layer", fontsize = 14)
    plt.ylabel("Hidden layers", fontsize = 14)
    plt.xticks(size = 13)
    plt.yticks(size = 13)
    # plt.title(f"Neural Network Architecture vs MSE {each}")
    plt.savefig(f"figures/c)/adam_heat_map_{each}.pdf")
    plt.show()

    # Find the best architecture
    best = np.unravel_index(np.argmin(mse), mse.shape)
    print(f"Best hidden layers for {each}:", n_hidden_layers[best[0]])
    print(f"Best hidden nodes for {each}:", number_of_nodes[best[1]])
    print(f"Lowest MSE for {each}:", mse[best])

