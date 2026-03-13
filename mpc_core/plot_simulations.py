import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def plot_policy_montecarlo_simulation(model, lower_bound_x, upper_bound_x, SIMULATION_HORIZON, NUMBER_EXPERIMENTS_MONTECARLO, simulation_list_policy_state, lower_bounds, all_vertices, centers, goal_centers, critical_centers_indexes, cell_width, PLOT_SIMULATION):
    
    if PLOT_SIMULATION != "True":
        return

    # Set plotting parameters
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 14
    })

    if model == 'Double_integrator' or model == 'Dubins_small':
   

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

    elif model == 'Mountain_car':

        fig, ax = plt.subplots()

        # ax.set_title(f"MPC interface simulation {SIMULATION_ID}")

        # plt.scatter(x[:,0], x[:,1],zorder = 3, label="MPC")



        # for k in range(SIMULATION_HORIZON):
        #     plt.plot([x[k,0],x[k+1,0]], [x[k,1],x[k+1,1]], 'blue',linewidth=1)
        for i in range(NUMBER_EXPERIMENTS_MONTECARLO):
            x_trajectory = simulation_list_policy_state[i]
            # plt.scatter(x_trajectory[:,0], x_trajectory[:,1],zorder = 3, color='red', label="Policy")
            for k in range(SIMULATION_HORIZON):
                # ax.scatter(x_trajectory[:,0], x_trajectory[:,1],zorder = 3, color='red', label="Policy")
                ax.plot([x_trajectory[k,0],x_trajectory[k+1,0]], [x_trajectory[k,1],x_trajectory[k+1,1]], 'red',linewidth=1)

        for i in range(NUMBER_EXPERIMENTS_MONTECARLO):
            x_trajectory = simulation_list_policy_state[i]
            # plt.scatter(x_trajectory[:,0], x_trajectory[:,1],zorder = 3, color='red', label="Policy")
            for k in range(SIMULATION_HORIZON):
                # ax.scatter(x_trajectory[:,0], x_trajectory[:,1],zorder = 3, color='red', label="Policy")
                ax.plot([x_trajectory[k,0],x_trajectory[k+1,0]], [x_trajectory[k,1],x_trajectory[k+1,1]], 'green',linewidth=1)

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
        plt.xlim(lower_bound_x[0], upper_bound_x[0])
        plt.ylim(lower_bound_x[1], upper_bound_x[1])
        plt.xlabel("Position")
        plt.ylabel("Velocity")

        # plt.legend(
        #     loc="lower right",
        #     bbox_to_anchor=(1, 0),     
        #     borderaxespad=0,
        #     fancybox=False,      
        #     edgecolor="black"
        # )

        # plt.xlim([box_lower_bound[0],box_upper_bound[0]])
        # plt.ylim([box_lower_bound[1],box_upper_bound[1]])
        # ax.set_aspect("equal", adjustable="box")
        pbar.close()

        plt.tight_layout()

    plt.show()
    