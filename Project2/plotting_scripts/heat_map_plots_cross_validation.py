import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


"""
LLM tag: ChatGPT wrote the sceleton of this short plotting code; we have changed it to work exactly how we wanted it to. 
The code snippet for modifying the colour map was written by ChatGPT and checked by us to make sure it worked as we wanted to.  
"""

lambda_values = np.concatenate(([0.0], np.logspace(-5, -3, 10)))
print(lambda_values)

method = "rmsprop"

# Original colormap
cmap = plt.cm.RdPu_r

# Remove the lightest 15% of the color scale
new_colors = cmap(np.linspace(0.1, 1, 256))

# Create a new colormap
new_cmap = mcolors.LinearSegmentedColormap.from_list(
    'RdPu_darker', new_colors
)


mse = np.loadtxt(f"data/mse_crossvalidation_{method}_l1.txt")


gamma_values = [10**-3, 10**-2, 5*10**-2, 10**-1, 5*10**-1]
lambda_values = [0, 10**-5, 10**-4, 5*10**-4, 10**-3]


annotations = np.where(mse < 0.099999, np.vectorize(lambda x: f"{x:.3f}")(mse),"")
ax = sns.heatmap(mse, annot = annotations, annot_kws={"size": 12}, fmt="", cmap = new_cmap, xticklabels = lambda_values, yticklabels = gamma_values)


cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)
plt.xlabel(rf"$\lambda$", fontsize = 16)
plt.ylabel(rf"$\gamma$", fontsize = 16)
yticks = [10**-3, 10**-2, 5*10**-2, 10**-1, 5*10**-1]
plt.yticks(np.arange(len(yticks)) + 0.5,[f"{y:.3f}" for y in yticks], size=14)
xticks = [0, 10**-5, 10**-4, 5*10**-4, 10**-3]
plt.xticks(np.arange(len(xticks)) + 0.5,[f"{x:.4f}" for x in xticks], size=14)
# plt.title(f"Neural Network Architecture vs MSE {each}")
plt.savefig(f"figures/f)/crossvalidation_heatmap_{method}.pdf")
plt.show()

# Find the best architecture
best = np.unravel_index(np.argmin(mse), mse.shape)
print(f"Best gamma:", gamma_values[best[0]])
print(f"Best lambda:", lambda_values[best[1]])
print(f"Lowest MSE for:", mse[best])

