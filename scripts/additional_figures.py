# %%
import numpy as np
import matplotlib.pyplot as plt

figures_path = '../results/plots/'
# %%

# a roving paradigm illustration

y = np.array([
    3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1, 1, 
    7,7,7,7,7,7,7,7,
    2, 2, 2, 2,2,2,
    4, 4, 4, 4, 4, 4, 4, 4, 4,

]) * 50 + 450
x = np.linspace(0, .6 * len(y), len(y))
diffs = np.diff(y) != 0
devs = np.append([False], diffs)

devs
# %%

fig, axs = plt.subplots(figsize=(7, 4))
axs.scatter(x[devs], y[devs], c='red', label='deviant', marker='>')
axs.scatter(x[~devs], y[~devs], c='blue', label='standard', marker='>')
axs.legend()
axs.set_xlabel('Time (s)')
axs.set_ylabel('Fundamental frequency (Hz)')
fig.savefig(figures_path + 'roving_example.png', dpi=300)

# %%
