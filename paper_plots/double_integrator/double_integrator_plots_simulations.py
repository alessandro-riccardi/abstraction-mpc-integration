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

# %% MARK: Settings

SIMULATION_ID = "07" # the simulation you want to plot

SHOW_PLOTS = False

NUMBER_EXPERIMENTS_MPC_PLOT = int(20) # Number of MPC simulations to plot, must be less than or equal to NUMBER_EXPERIMENTS_MPC

LEGEND_FONTSIZE = 20
PLOT_FONTSIZE = 30

import matplotlib.pyplot as plt
# Set plotting parameters
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": PLOT_FONTSIZE
})

# %% MARK: Load Files

current_dir = os.getcwd()


file_path = os.path.join(current_dir, "mpc_simulation_data", "mpc_simulation_double_integrator", f"simulation_mpc_double_integrator_{SIMULATION_ID}.pkl")
with open(file_path, "rb") as f:   
    simulation_data_MPC = pickle.load(f)

file_path = os.path.join(current_dir, "abstraction_data", "abstraction_data_double_integrator", "abstraction_data_DoubleIntegrator_01.pkl")
with open(file_path, "rb") as f:   
    abstraction_data = pickle.load(f)

NUMBER_EXPERIMENTS_MPC = simulation_data_MPC["NUMBER_EXPERIMENTS_MPC"]
simulation_list_policy_state = simulation_data_MPC["simulation_list_policy_state"]
simulation_list_MPC_state = simulation_data_MPC["simulation_list_MPC_state"] 
lower_bound_x = simulation_data_MPC["lower_bound_x"]
upper_bound_x = simulation_data_MPC["upper_bound_x"]

SIMULATION_HORIZON = len(simulation_list_policy_state[0])-1

cell_width = abstraction_data['cell_width']
lower_bounds = abstraction_data['lower_bounds']
all_vertices = abstraction_data['all_vertices']
centers = abstraction_data['centers']
goal_centers = abstraction_data['goal_centers']
critical_centers_indexes = abstraction_data['critical_centers_indexes']


CELLS_NUMBER = lower_bounds.shape[0]
squares_corners = all_vertices[:,0,0:2]
squares_corners = np.unique(squares_corners, axis=0)




# %% MARK: Plot behaviour

fig, ax = plt.subplots()
# plt.rcParams.update({'font.size': 22})


ALPHA = 1

for i in range(NUMBER_EXPERIMENTS_MPC):
    x_trajectory = simulation_list_policy_state[i]
    # if i == 0:
        # handles +x_trajectory[0,:])
    # plt.scatter(x_trajectory[:,0], x_trajectory[:,1],zorder = 3, color='red', label="Policy")

    for k in range(SIMULATION_HORIZON):
        if (i ==0) and (k == 0):
            plt.plot([x_trajectory[k,0],x_trajectory[k+1,0]], [x_trajectory[k,1],x_trajectory[k+1,1]],  'red' ,alpha= ALPHA, linewidth=1, label='Policy')
        else:
            plt.plot([x_trajectory[k,0],x_trajectory[k+1,0]], [x_trajectory[k,1],x_trajectory[k+1,1]], 'red', alpha= ALPHA,linewidth=1, label='_nolegend_')

for i in range(NUMBER_EXPERIMENTS_MPC):
    x_trajectory = simulation_list_MPC_state[i]
    for k in range(SIMULATION_HORIZON):
        if (i ==0) and (k == 0):
            plt.plot([x_trajectory[k,0],x_trajectory[k+1,0]], [x_trajectory[k,1],x_trajectory[k+1,1]], 'blue', alpha= ALPHA,linewidth=1, label='MPC')
        else:
            plt.plot([x_trajectory[k,0],x_trajectory[k+1,0]], [x_trajectory[k,1],x_trajectory[k+1,1]], 'blue', alpha= ALPHA,linewidth=1, label='_nolegend_')
# Manually define handles and labels for the legend
# labels = ['MPC', 'Policy']
# Create the legend using the defined handles and labels
legend = plt.legend(loc='upper right', frameon=True, shadow=False, fontsize = LEGEND_FONTSIZE)
legend.get_frame().set_boxstyle('square')
frame = legend.get_frame()
frame.set_edgecolor('black')

CELLS_NUMBER = lower_bounds.shape[0]
squares_corners = all_vertices[:,0,0:2]
squares_corners = np.unique(squares_corners, axis=0)

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


for i in range(CELLS_NUMBER):
    for j in range(len(goal_centers)):
        if (centers[i,:] == goal_centers[j,:]).all():
        
            # print(f"Center: {centers[i,:]}")
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
        square = plt.Rectangle(
            (centers[i,0]-(delta_x/2), centers[i,1]-(delta_x/2)), delta_x, delta_x,
            facecolor="black",
            alpha = 1,   
            edgecolor="black",
            linewidth=0.5,
            zorder = 1
        )
        ax.add_patch(square)
    pbar.update(1)


ax.set_aspect("equal", adjustable="box")
plt.xlim(lower_bound_x[0], upper_bound_x[0])
plt.ylim(lower_bound_x[1], upper_bound_x[1])
plt.xlabel(r"State $x^{[1]}$")
plt.ylabel(r"State $x^{[2]}$")
pbar.close()


plt.tight_layout()

if SHOW_PLOTS:
    plt.show()


file_path = os.path.join(current_dir, "paper_plots", "double_integrator", f"simulation_double_integrator_{SIMULATION_ID}_plot.pdf")
fig.savefig(file_path, bbox_inches="tight", dpi = 300)
file_path = os.path.join(current_dir, "paper_plots", "double_integrator", f"simulation_double_integrator_{SIMULATION_ID}_plot.png")
fig.savefig(file_path, bbox_inches="tight", dpi = 300)