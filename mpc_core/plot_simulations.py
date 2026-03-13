import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def plot_policy_montecarlo_simulation(SIMULATION_HORIZON, NUMBER_EXPERIMENTS_MONTECARLO, simulation_list_policy_state, lower_bounds, all_vertices, centers, goal_centers, critical_centers_indexes, cell_width, PLOT_SIMULATION):
    fig, ax = plt.subplots()


    for i in range(NUMBER_EXPERIMENTS_MONTECARLO):
        x_trajectory = simulation_list_policy_state[i]
        for k in range(SIMULATION_HORIZON):
            ax.plot([x_trajectory[k,0],x_trajectory[k+1,0]], [x_trajectory[k,1],x_trajectory[k+1,1]], 'red',linewidth=1)

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
    pbar.close()

    plt.tight_layout()

    if PLOT_SIMULATION == True:
        plt.show()