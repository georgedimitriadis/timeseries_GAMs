
from typing import List
from numpy._typing import NDArray
import numpy as np
from matplotlib import pyplot as plt
import matplotlib


matplotlib.rcParams.update({'font.size': 8})

def show_features_distributions(raw_x_data, features_names, bins=50):
    num_of_features = len(features_names)
    cols = 3
    rows = int(np.ceil(num_of_features / cols))
    f = plt.figure()
    hist_axs = {}
    c = r = 0
    for a in range(num_of_features):
        hist_axs[features_names[a]] = plt.subplot2grid((rows, cols), (r, c), rowspan=1, colspan=1, fig=f)
        hist_axs[features_names[a]].hist(raw_x_data[:, a], bins=bins)
        hist_axs[features_names[a]].set_title(features_names[a])
        c += 1
        if c == cols:
            c = 0
            r += 1
    return f, hist_axs

def show_2d_combinations(data: List[NDArray], features_names: List[str]):
    num_of_features = len(features_names)

    num_of_combinations = len(np.triu_indices(num_of_features, 1)[0])
    combination_indices = np.array(list(zip(np.triu_indices(num_of_features, 1)[0], np.triu_indices(num_of_features, 1)[1])))
    combination_names = [f'{features_names[a]} x {features_names[b]}' for a, b in zip(combination_indices[:, 0], combination_indices[:, 1])]
    cols = 4
    rows = int(np.ceil(num_of_combinations / cols))
    f = plt.figure()
    axs = {}
    c = r = 0
    for a in range(len(combination_names)):
        axs[combination_names[a]] = plt.subplot2grid((rows, cols), (r, c), rowspan=1, colspan=1, fig=f)
        for d in data:
            axs[combination_names[a]].scatter(d[:, combination_indices[a, 0]], d[:, combination_indices[a, 1]], s=5)
        axs[combination_names[a]].set_title(combination_names[a])
        c += 1
        if c == cols:
            c = 0
            r += 1
    f.tight_layout()
    f.subplots_adjust(wspace=0.2, hspace=0.2, left=0.04, bottom=0.04, top=0.94, right=0.94)


def add_stat_annotation(ax, x1, x2, y, p_value, h=0.05):
    """Add significance bracket between two groups"""
    # Determine significance level
    if p_value < 0.001:
        sig_symbol = '***'
    elif p_value < 0.01:
        sig_symbol = '**'
    elif p_value < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = 'ns'  # not significant
    # Draw the bracket
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='black')
    # Add the significance symbol
    ax.text((x1 + x2) / 2, y + h, sig_symbol, ha='center', va='bottom', fontsize=12)