import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os


def plot_mpc_montecarlo_simulation(PLOT_SIMULATION, STORE_SIMULATION_DATA, model, NUMBER_EXPERIMENTS_MPC, SIMULATION_HORIZON, simulation_list_policy_state, simulation_list_MPC_state, lower_bounds, all_vertices, centers, goal_centers, critical_centers_indexes, cell_width, lower_bound_x, upper_bound_x, current_dir, SIMULATION_ID):
    
    if PLOT_SIMULATION != "True":
        return

    if model == 'Double_integrator':
        mpc_simulation_folder = "mpc_simulation_double_integrator"
    elif model == 'Mountain_car':
        mpc_simulation_folder = "mpc_simulation_mountain_car"
    elif model == 'Dubins_small':
        mpc_simulation_folder = "mpc_simulation_small_dubin"

    CELLS_NUMBER = lower_bounds.shape[0]
    squares_corners = all_vertices[:,0,0:2]
    squares_corners = np.unique(squares_corners, axis=0)
        
    fig, ax = plt.subplots()

    ALPHA = 1

    NUMBER_EXPERIMENTS_MPC_PLOT = int(np.floor(NUMBER_EXPERIMENTS_MPC))

    for i in range(NUMBER_EXPERIMENTS_MPC_PLOT):
        x_trajectory = simulation_list_policy_state[i]
        
        for k in range(SIMULATION_HORIZON):
            if (i ==0) and (k == 0):
                plt.plot([x_trajectory[k,0],x_trajectory[k+1,0]], [x_trajectory[k,1],x_trajectory[k+1,1]],  'red' ,alpha= ALPHA, linewidth=1, label='Policy')
            else:
                plt.plot([x_trajectory[k,0],x_trajectory[k+1,0]], [x_trajectory[k,1],x_trajectory[k+1,1]], 'red', alpha= ALPHA,linewidth=1, label='_nolegend_')

    for i in range(NUMBER_EXPERIMENTS_MPC_PLOT):
        x_trajectory = simulation_list_MPC_state[i]

        for k in range(SIMULATION_HORIZON):
            if (i ==0) and (k == 0):
                plt.plot([x_trajectory[k,0],x_trajectory[k+1,0]], [x_trajectory[k,1],x_trajectory[k+1,1]], 'blue', alpha= ALPHA,linewidth=1, label='MPC')
            else:
                plt.plot([x_trajectory[k,0],x_trajectory[k+1,0]], [x_trajectory[k,1],x_trajectory[k+1,1]], 'blue', alpha= ALPHA,linewidth=1, label='_nolegend_')

    # Create the legend using the defined handles and labels
    legend = plt.legend(loc='upper right', frameon=True, shadow=False, fontsize=17)
    legend.get_frame().set_boxstyle('square')
    frame = legend.get_frame()
    frame.set_edgecolor('black')

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


    if model == 'Double_integrator':
        plt.xlabel(r"$x^{[1]}$", fontsize=21)
        plt.ylabel(r"$x^{[2]}$", fontsize=21)
        ax.set_aspect("equal", adjustable="box")
    elif model == 'Mountain_car':
        plt.xlabel("Position", fontsize=21)
        plt.ylabel("Velocity", fontsize=21)
    elif model == 'Dubins_small':
        plt.xlabel("Position x", fontsize=21)
        plt.ylabel("Position y", fontsize=21)
        ax.set_aspect("equal", adjustable="box")

    
    pbar.close()

    plt.tight_layout()

    plt.show()

    if STORE_SIMULATION_DATA == "True":
        file_path = os.path.join(current_dir, "mpc_simulation_data", mpc_simulation_folder, f"simulation_{SIMULATION_ID}_plot.pdf")
        fig.savefig(file_path, dpi = 300, bbox_inches="tight")           
        
    


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
    