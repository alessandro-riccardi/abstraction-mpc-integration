import numpy as np
import time
from tqdm import tqdm
from mpc_core.Double_integrator_dynamics import * # make lower case, modify file
from mpc_core.mountain_car_dynamics import *
from mpc_core.Dubins_small_dynamics import * # make lower case, modify file
from mpc_core.mpc_support_functions import get_cell_index


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