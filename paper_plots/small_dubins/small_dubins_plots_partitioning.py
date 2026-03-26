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
    "font.size": 14
})

SIMULATION_ID = "00"
LOAD_MODEL = False
MAKE_PLOT = True
SHOW_PLOT = True
STORE_PLOT = False
STORE_DATA = False


# %% MARK: Load Simulation Files

# Where this script is located
current_dir = os.getcwd()


# import core

# Get the path of the required file
file_path = os.path.normpath(os.path.join(current_dir, "abstraction_data", "abstraction_data_small_dubin", "abstraction_data_Dubins_small_04.pkl"))
file_path_nominal = os.path.normpath(os.path.join(current_dir,  "abstraction_data", "abstraction_data_small_dubin", "abstraction_data_Dubins_small_01.pkl"))


# Load pickle file
with open(file_path, "rb") as f:   
    simulation_data = pickle.load(f)

with open(file_path_nominal, "rb") as f:   
    simulation_data_nominal = pickle.load(f)

# Acquire abstraction data

# partition = simulation_data.partition
# regions = simulation_data.regions
# actions = simulation_data.actions
policy_inputs = simulation_data['policy_inputs']
actions_inputs = simulation_data['actions.inputs']
# actions_inputs = simulation_data['actions_inputs']
upper_bounds = simulation_data['upper_bounds']
lower_bounds = simulation_data['lower_bounds']
all_vertices = simulation_data['all_vertices']
centers = simulation_data['centers']
goal_centers = simulation_data['goal_centers']
critical_centers = simulation_data['critical_centers']
critical_centers_indexes = simulation_data['critical_centers_indexes']
policy = simulation_data['policy']
cell_width = simulation_data['cell_width']
Lp_balls = simulation_data['epsilons']

policy_inputs_nominal = simulation_data_nominal['policy_inputs']
actions_inputs_nominal = simulation_data_nominal['actions.inputs']
# actions_inputs_nominal = simulation_data_nominal['actions_inputs']
policy_nominal = simulation_data_nominal['policy']


# %% MARK: Plot State Space 


CELLS_NUMBER = lower_bounds.shape[0]
squares_corners = all_vertices[:,0,0:2]
squares_corners = np.unique(squares_corners, axis=0)


box_upper_bound = np.max(upper_bounds,axis=0)
box_lower_bound = np.min(lower_bounds,axis=0)

ANGLE_MULTIPLIER = 1
lower_bound_x = np.array([box_lower_bound[0], box_lower_bound[1], ANGLE_MULTIPLIER*box_lower_bound[2]]).astype(np.float64)
upper_bound_x = np.array([box_upper_bound[0], box_upper_bound[1], ANGLE_MULTIPLIER*box_upper_bound[2]]).astype(np.float64)

x0 = simulation_data['initial_state']

fig, ax = plt.subplots()

delta_x = cell_width[0]

pbar = tqdm(total=len(squares_corners)+CELLS_NUMBER, desc="Contructing figure")

for i in range(len(squares_corners)):
    square = plt.Rectangle(
        (squares_corners[i,0], squares_corners[i,1]), delta_x, delta_x,
        fill=False,   
        edgecolor="lightgrey",
        linewidth=0.5,
        linestyle="--",
        zorder = 0
    )
    ax.add_patch(square)
    pbar.update(1)


labeled_target=False
labeled_avoid=False

ax.plot(x0[0], x0[1], 'ro', markersize=7, label='Starting Location')


for i in range(CELLS_NUMBER):
    for j in range(len(goal_centers)):
        if (centers[i,:] == goal_centers[j,:]).all():
        
            # print(f"Center: {centers[i,:]}")
            if labeled_target == False:
                square = plt.Rectangle(
                    (all_vertices[i,0,0], all_vertices[i,0,1]), delta_x, delta_x,
                    facecolor="cyan",
                    alpha = 0.2,   
                    edgecolor="lightgrey",
                    linewidth=0.5,
                    linestyle="--",
                    zorder = 1,
                    label='Target area'
                )
                ax.add_patch(square)
                labeled_target = True
            else:
                square = plt.Rectangle(
                    (all_vertices[i,0,0], all_vertices[i,0,1]), delta_x, delta_x,
                    facecolor="cyan",
                    alpha = 0.2,   
                    edgecolor="lightgrey",
                    linewidth=0.5,
                    linestyle="--",
                    zorder = 1
                )
                ax.add_patch(square)

    if np.isin(i,critical_centers_indexes).all():
        if labeled_avoid == False:
            square = plt.Rectangle(
                (centers[i,0]-(delta_x/2), centers[i,1]-(delta_x/2)), delta_x, delta_x,
                facecolor="black",
                alpha = 0.15,   
                edgecolor="black",
                linewidth=0.5,
                zorder = 1,
                label='Area to avoid'
            )
            ax.add_patch(square)
            labeled_avoid = True
        else:
            square = plt.Rectangle(
                (centers[i,0]-(delta_x/2), centers[i,1]-(delta_x/2)), delta_x, delta_x,
                facecolor="grey",
                alpha = 0.15,  
                edgecolor="grey",
                linewidth=0.5,
                zorder = 1
            )
            ax.add_patch(square)
    pbar.update(1)



legend = plt.legend(loc='upper right', frameon=True, shadow=False, fontsize=17)
legend.get_frame().set_boxstyle('square')
frame = legend.get_frame()
frame.set_edgecolor('black')


plt.xlim(lower_bound_x[0], upper_bound_x[0])
plt.ylim(lower_bound_x[1], upper_bound_x[1])
plt.xlabel("Position x", fontsize=21)
plt.ylabel("Position y", fontsize=21)
ax.set_aspect("equal", adjustable="box")
pbar.close()

plt.tight_layout()
plt.show()


file_path = os.path.join(current_dir, "paper_plots", "small_dubins", f"small_dubins_statespace.pdf")
fig.savefig(file_path, bbox_inches="tight", dpi = 300)


# %% MARK: Plot Input Space

upper_bound_u = np.max(policy_inputs, axis=0).astype(np.float64)
lower_bound_u = np.min(policy_inputs, axis=0).astype(np.float64)


fig, ax = plt.subplots()

width  = upper_bound_u[0] - lower_bound_u[0]
height = upper_bound_u[1] - lower_bound_u[1]

# Create rectangle: xy=lower-left corner, edgecolor='black', no fill
rect = patches.Rectangle(
    xy        = (lower_bound_u[0], lower_bound_u[1]),
    width     = width,
    height    = height,
    linewidth = 1,
    edgecolor = 'black',
    facecolor = 'lightgreen',
    alpha = 0.3
)

ax.add_patch(rect)



for action_idx in range(0, len(actions_inputs)):
    ax.plot(actions_inputs[action_idx,0], actions_inputs[action_idx,1], 'ro', markersize=2.5)
    corner_1 = [np.max([actions_inputs[action_idx,0]-Lp_balls[0]/2,lower_bound_u[0]]),np.max([actions_inputs[action_idx,1]-Lp_balls[1]/2,lower_bound_u[1]])]
    corner_2 = [np.min([actions_inputs[action_idx,0]+Lp_balls[0]/2,upper_bound_u[0]]),np.max([actions_inputs[action_idx,1]-Lp_balls[1]/2,lower_bound_u[1]])]
    corner_3 = [np.min([actions_inputs[action_idx,0]+Lp_balls[0]/2,upper_bound_u[0]]),np.min([actions_inputs[action_idx,1]+Lp_balls[1]/2,upper_bound_u[1]])]
    corner_4 = [np.max([actions_inputs[action_idx,0]-Lp_balls[0]/2,lower_bound_u[0]]),np.min([actions_inputs[action_idx,1]+Lp_balls[1]/2,upper_bound_u[1]])]
    
    corners = [corner_1, corner_2, corner_3, corner_4]
    polygon = Polygon(
        corners,
        linewidth = 1,
        edgecolor = 'cyan',
        facecolor = 'cyan',
        alpha = 0.7
    )
    ax.add_patch(polygon)

plt.xlabel(r"Control action $u_1$", fontsize=21)
plt.ylabel(r"Control action $u_2$", fontsize=21)


ax.set_aspect("equal", adjustable="box")

plt.xlim(lower_bound_u[0]*1.1, upper_bound_u[0]*1.1)
plt.ylim(lower_bound_u[1]*1.1, upper_bound_u[1]*1.1)

plt.tight_layout()
plt.show()

file_path = os.path.join(current_dir, "paper_plots", "small_dubins", f"small_dubins_inputspace.pdf")
fig.savefig(file_path, bbox_inches="tight", dpi = 300)
# %%
