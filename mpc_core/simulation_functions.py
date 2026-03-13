import numpy as np
import time
from tqdm import tqdm
from mpc_core.Double_integrator_dynamics import * 
from mpc_core.mountain_car_dynamics import *
from mpc_core.Dubins_small_dynamics import * 
from mpc_core.mpc_support_functions import get_cell_index

def mpc_montecarlo_simulation():
    start_time = time.perf_counter()
    # pbar_MPC = tqdm(total=NUMBER_EXPERIMENTS_MPC, desc="Solving MPC reconstruction problem")
    print(f"Solving MPC reconstruction problem")
    violation_counter = 0
    backup_policy_usage_counter = 0
    simulation_steps_counter = 0
    for i in range(NUMBER_EXPERIMENTS_MPC):

        

        start_time_experiment = time.perf_counter()
        x_trajectory = np.zeros((SIMULATION_HORIZON+1+PREDICTION_HORIZON, STATES_NUMBER))
        u_trajectory = np.zeros((SIMULATION_HORIZON+1+PREDICTION_HORIZON, INPUTS_NUMBER))
        w_sequence = simulation_list_noise[i]

        # x_trajectory[0,:] = x0
        x_trajectory[0,:] = x0 + simulation_list_initial_state[i].copy()
        no_action = np.array([0,0])

        dubins_real_MPC = DubinsSmallDynamicsStochastic(x0 + simulation_list_initial_state[i].copy())
        cost_policy_i = 0
        cost_policy_state_i = 0
        cost_policy_input_i = 0
        for k in range(SIMULATION_HORIZON):
            print(f"MPC Experiment {i+1}/{NUMBER_EXPERIMENTS_MPC}")
            print(f"Time step {k}/{SIMULATION_HORIZON}")
            x_k = x_trajectory[k,:]
            initial_cell_idx = get_cell_index(centers, x_k)

            action_index = policy[initial_cell_idx]

            if (action_index == -1):
                u_trajectory[k,:] = no_action
                x_trajectory[k+1:SIMULATION_HORIZON+PREDICTION_HORIZON+1,:] = x_trajectory[k,:]
                break
            # if (action_index == -1) and (np.isin(initial_cell_idx, goal_centers_indices) == True):
            #     u_trajectory[k,:] = no_action
            #     x_trajectory[k+1:SIMULATION_HORIZON+PREDICTION_HORIZON+1,:] = x_trajectory[k,:]
            #     break
            # elif (action_index == -1) and (np.isin(initial_cell_idx, goal_centers_indices) == False):
            #     # u_k = no_action
            #     u_trajectory[k,:] = no_action
            #     x_trajectory[k+1:SIMULATION_HORIZON+PREDICTION_HORIZON+1,:] = x_trajectory[k,:]
            #     violation_counter += 1
            #     print(f"Specification violation occurred: break")
            #     print(f"Current number of violations: {violation_counter}")
            #     break
            # elif (x_k < lower_bound_x).any() or (x_k > upper_bound_x).any():
            #     u_trajectory[k,:] = no_action
            #     x_trajectory[k+1:SIMULATION_HORIZON+PREDICTION_HORIZON+1,:] = x_trajectory[k,:]
            #     violation_counter += 1
            #     print(f"Specification violation occurred, boundary crossed: break")
            #     print(f"Current number of violations: {violation_counter}")
            #     break
            else:

            # if action_index == -1:
            #     # u_k = no_action
            #     u_trajectory[k,:] = no_action
            #     x_trajectory[k+1:SIMULATION_HORIZON+PREDICTION_HORIZON+1,:] = x_trajectory[k,:]
            #     break
            # else:
                # Run MPC

                # Create a local copy of the general MPC problem
                MPC_model_k = MPC_model.copy()

                # Set up initial condition constraints
                MPC_model_k.addConstr(x_tilde_k[0,:] == x_k)

                # Assign first boolean variable
                for cell_idx in range(0,Ns):
                    if cell_idx == initial_cell_idx:
                        MPC_model_k.addConstr(delta_s_k[0,cell_idx] == True)
                    else: 
                        MPC_model_k.addConstr(delta_s_k[0,cell_idx] == False)

                # Select target cell at time step k
                target_cell_k = target_cell[initial_cell_idx]
                # target_cell_k = target_cell_policy[initial_cell_idx]

                cost = 0
                # Construct cost function at time step k
                for j in range(PREDICTION_HORIZON):
                    cost += u_tilde_k[j,:] @ R @ u_tilde_k[j,:] 
                    cost += (target_cell_k - x_tilde_k[j+1,:]) @ Q @ (target_cell_k - x_tilde_k[j+1,:])
                    # cost += (x_tilde_k[j+1,:] - x_tilde_k[j,:]) @ Q @ (x_tilde_k[j+1,:] - x_tilde_k[j,:])
                
                MPC_model_k.setObjective(cost, GRB.MINIMIZE)

                # Solve local MPC problem
                MPC_model_k.optimize()

                if MPC_model_k.Status == GRB.INFEASIBLE:
                    print(f"INFEASIBLE PROBLEM")
                    initial_cell_idx = get_cell_index(centers, x_trajectory[k,:])

                    action_index = policy[initial_cell_idx]
                    u_trajectory[k,:] = actions_inputs[action_index,:]
                    backup_policy_usage_counter += 1
                    print(f"Deploy backup policy")
                    # x_trajectory[k+1:SIMULATION_HORIZON+PREDICTION_HORIZON+1,:] = x_trajectory[k,:]
                    # violation_counter += 1
                    # break
                else:
                    # Retrieving the control action
                    opt_solution = MPC_model_k.getVars()[0:INPUTS_NUMBER]
                    for j in range(INPUTS_NUMBER):
                        u_trajectory[k,j] = opt_solution[j].X

                # Applying the control to the real system
                
                x_trajectory[k+1,:] = dubins_real_MPC.step(u_trajectory[k,:],w_sequence[k])
                # x_trajectory[k+1,2] = (x_trajectory[k+1,2] + np.pi) % (2 * np.pi) - np.pi
                # cost_policy_i += u_trajectory[k,:] @ R @ u_trajectory[k,:] + (target_cell_k - x_trajectory[k+1,:])@Q@(target_cell_k - x_trajectory[k+1,:])
                # cost_policy_i += u_trajectory[k,:] @ R @ u_trajectory[k,:] + (x_trajectory[k+1,:] - x_trajectory[k,:])@Q@(x_trajectory[k+1,:] - x_trajectory[k,:])
                cost_policy_state_i += (target_cell_k - x_trajectory[k+1,:])@Q@(target_cell_k - x_trajectory[k+1,:])
                # cost_policy_state_i += (x_trajectory[k+1,:] - x_trajectory[k,:])@Q@(x_trajectory[k+1,:] - x_trajectory[k,:])
                # cost_policy_state_i += (x_trajectory[k+PREDICTION_HORIZON,:] - x_trajectory[k,:])@Q@(x_trajectory[k+PREDICTION_HORIZON,:] - x_trajectory[k,:])
                cost_policy_input_i += u_trajectory[k,:] @ R @ u_trajectory[k,:] 
                cost_policy_i += (target_cell_k - x_trajectory[k+1,:])@Q@(target_cell_k - x_trajectory[k+1,:]) + u_trajectory[k,:] @ R @ u_trajectory[k,:] 

                MPC_model_k.dispose()
                simulation_steps_counter += 1
            comulative_cost_MPC[i] = cost_policy_i
            comulative_cost_MPC_state[i] = cost_policy_state_i
            comulative_cost_MPC_input[i] = cost_policy_input_i

        # Store simulation      
        simulation_list_MPC_state.append(x_trajectory)
        simulation_list_MPC_input.append(u_trajectory)

        elapsed_time_experiment = time.perf_counter() - start_time_experiment
        print(f"MPC simulation {i} required: {elapsed_time_experiment:.6f} seconds")
        print(f"Total enlapsed time: {(time.perf_counter() - start_time):.6f} seconds")

        # TEMPORARY PLOT
        # for k in range(SIMULATION_HORIZON):
        #     ax.plot([x_trajectory[k,0],x_trajectory[k+1,0]], [x_trajectory[k,1],x_trajectory[k+1,1]], 'blue',linewidth=1)
        # if SHOW_PLOT == True:
        #     clear_output(wait=True)
        #     display(fig)

        # pbar_MPC.update(1)
    # pbar_MPC.close()

    elapsed_time = time.perf_counter() - start_time
    print(f"MPC Montecarlo simulation time: {elapsed_time:.6f} seconds")
    print(f"Average MPC simulation time: {(elapsed_time/NUMBER_EXPERIMENTS_MPC):.6f} seconds")
    # Compute empirical satisfaction probability
    satisfaction_counter = 0
    for i in range(NUMBER_EXPERIMENTS_MPC):
        x_trajectory = simulation_list_MPC_state[i].copy()
        final_cell_index = get_cell_index(centers, x_trajectory[-1])
        if np.isin(final_cell_index,goal_centers_indices) == True:
            satisfaction_counter += 1
    empirical_satisfaction_probability_MPC = (satisfaction_counter)/NUMBER_EXPERIMENTS_MPC

    elapsed_time = time.perf_counter() - start_time
    print(f"Policy Montecarlo simulation time: {elapsed_time:.6f} seconds")
    print(f"Empirical satisfaction probability MPC: {empirical_satisfaction_probability_MPC}")


def policy_montecarlo_simulation(SIMULATION_HORIZON, PREDICTION_HORIZON, NUMBER_EXPERIMENTS_MONTECARLO, 
                                 STATES_NUMBER, INPUTS_NUMBER, model, simulation_list_noise, x0, simulation_list_initial_state, 
                                 centers, goal_centers_indices, target_cell, policy_nominal, actions_inputs_nominal, Q, R, 
                                 lower_bound_x, upper_bound_x, cumulative_cost_policy, cumulative_cost_policy_state, 
                                 cumulative_cost_policy_input, simulation_list_policy_state, simulation_list_policy_input):
    

    pbar_MPC = tqdm(total=SIMULATION_HORIZON, desc="Montecarlo simulation of control policy")



    violation_counter = 0

    for i in range(NUMBER_EXPERIMENTS_MONTECARLO):

        x_trajectory = np.zeros((SIMULATION_HORIZON+1+PREDICTION_HORIZON, STATES_NUMBER))
        u_trajectory = np.zeros((SIMULATION_HORIZON+1+PREDICTION_HORIZON, INPUTS_NUMBER))
        w_sequence = simulation_list_noise[i]

        x_trajectory[0,:] = x0 + simulation_list_initial_state[i].copy()
        no_action = np.zeros(INPUTS_NUMBER)

        if model == 'Double_integrator':
            model_policy = DoubleIntegratorDynamics(x0 + simulation_list_initial_state[i].copy())
        elif model == 'Mountain_car':
            model_policy = MountainCarDynamics(x0 + simulation_list_initial_state[i].copy())
        elif model == 'Dubins_small':
            model_policy = DubinsSmallDynamicsStochastic(x0 + simulation_list_initial_state[i].copy())    

    
        cost_policy_i = 0
        cost_policy_state_i = 0 
        cost_policy_input_i = 0

        for k in range(SIMULATION_HORIZON):
            x_k = x_trajectory[k,:]
            initial_cell_index = get_cell_index(centers, x_k)

            action_index = policy_nominal[initial_cell_index]

            # Select target cell at time step k
            target_cell_k = target_cell[initial_cell_index]

            if (action_index == -1) and (np.isin(initial_cell_index, goal_centers_indices) == True):
                u_trajectory[k,:] = no_action
                x_trajectory[k+1:SIMULATION_HORIZON+PREDICTION_HORIZON+1,:] = x_trajectory[k,:]
                break

            elif (action_index == -1) and (np.isin(initial_cell_index, goal_centers_indices) == False):
                
                u_trajectory[k,:] = no_action
                x_trajectory[k+1:SIMULATION_HORIZON+PREDICTION_HORIZON+1,:] = x_trajectory[k,:]
                violation_counter += 1
                print(f"Specification violation occurred: break")
                print(f"Current number of violations: {violation_counter}")
                break

            elif (x_k < lower_bound_x).any() or (x_k > upper_bound_x).any():
                u_trajectory[k,:] = no_action
                x_trajectory[k+1:SIMULATION_HORIZON+PREDICTION_HORIZON+1,:] = x_trajectory[k,:]
                violation_counter += 1
                print(f"Specification violation occurred, boundary crossed: break")
                print(f"Current number of violations: {violation_counter}")
                break

            else:
                u_trajectory[k,:] = actions_inputs_nominal[action_index,:]
                x_trajectory[k+1,:] = model_policy.step(actions_inputs_nominal[action_index,:],w_sequence[k])

                cost_policy_state_i += (target_cell_k - x_trajectory[k+1,:])@Q@(target_cell_k - x_trajectory[k+1,:])
                cost_policy_input_i += u_trajectory[k,:] @ R @ u_trajectory[k,:]
                cost_policy_i += (target_cell_k - x_trajectory[k+1,:])@Q@(target_cell_k - x_trajectory[k+1,:]) + u_trajectory[k,:] @ R @ u_trajectory[k,:]

                
        cumulative_cost_policy[i] = cost_policy_i
        cumulative_cost_policy_state[i] = cost_policy_state_i
        cumulative_cost_policy_input[i] = cost_policy_input_i
        simulation_list_policy_state.append(x_trajectory)
        simulation_list_policy_input.append(u_trajectory)

        pbar_MPC.update(1)
    pbar_MPC.close()

    return cumulative_cost_policy, cumulative_cost_policy_state, cumulative_cost_policy_input, simulation_list_policy_state, simulation_list_policy_input, violation_counter