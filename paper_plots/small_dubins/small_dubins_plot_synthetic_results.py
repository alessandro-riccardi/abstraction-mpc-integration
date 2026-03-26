# %% MARK: Load Libraries

import numpy as np
import os
import sys
import pickle
from tqdm import tqdm
import gurobipy as gp
from gurobipy import GRB
import cvxpy as cp
import time

# import core

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
# Set plotting parameters
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 24
})

# %% MARK: Simulations data

ball_area_sat_prob = np.array([0, 0.02, 0.045, 0.0512, 0.0578, 0.0648, 0.08])
ball_area_performance = np.array([0, 0.02, 0.045, 0.0512, 0.0578, 0.0648, 0.08])

sat_prob = np.array([0.9963, 0.9945, 0.9913, 0.9141, 0.8959, 0.7405, 0.1197])
performance = np.array([1770.68, 1743.48, 1729.71, 1769.54, 1790.47, 1755.56, 1909.15])

# %% MARK: Plot data


fig, ax1 = plt.subplots(figsize=(9, 5))

# Left axis — satisfaction probability
color_sat = '#3266ad'
ax1.set_xlabel('Ball area', size=24)
ax1.set_ylabel('Probability', color=color_sat, size=24)
ax1.plot(ball_area_sat_prob, sat_prob, color=color_sat, marker='o',
         linewidth=2, markersize=6, label=r'Satisfaction probability $\lambda$')
ax1.tick_params(axis='y', labelcolor=color_sat)
ax1.set_ylim(0, 1.05)

# Right axis — performance
color_perf = '#c0533a'
ax2 = ax1.twinx()
ax2.set_ylabel('Cost value', color=color_perf, size=24)
ax2.plot(ball_area_performance, performance, color=color_perf, marker='s',
         linestyle='--', linewidth=2, markersize=6, label=r'Average cost $J$')
ax2.tick_params(axis='y', labelcolor=color_perf)
ax2.set_ylim(1680, 1950)

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', frameon=False, shadow=False, fontsize=21)

ax1.grid(True, axis='both', color='lightgrey', linestyle='--', alpha=0.9)
ax2.grid(True, axis='both', color='lightgrey', linestyle='--', alpha=0.9)

legend.get_frame().set_boxstyle('square')
frame = legend.get_frame()
frame.set_edgecolor('black')



plt.tight_layout()

current_working_directory = os.getcwd()
file_path = os.path.join(current_working_directory, "paper_plots", "small_dubins", "dual_axis_plot.pdf")

plt.savefig(file_path, dpi=300)
plt.show()

# %%
