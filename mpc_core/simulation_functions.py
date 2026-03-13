import numpy as np
import time
from tqdm import tqdm
from mpc_core.Double_integrator_dynamics import * 
from mpc_core.mountain_car_dynamics import *
from mpc_core.Dubins_small_dynamics import * 
from mpc_core.mpc_support_functions import get_cell_index
import gurobipy as gp
from gurobipy import GRB


def mpc_montecarlo_simulation_double_integrator(NUMBER_EXPERIMENTS_MPC, SIMULATION_HORIZON, PREDICTION_HORIZON, STATES_NUMBER, INPUTS_NUMBER, simulation_list_noise, x0, upper_bound_u, lower_bound_u, upper_bound_x, lower_bound_x, simulation_list_initial_state, centers, goal_centers_indices, target_cell, policy, actions_inputs, R, Q, simulation_list_MPC_state, simulation_list_MPC_input, cumulative_cost_MPC, cumulative_cost_MPC_state, cumulative_cost_MPC_input, lb_z_s_k, ub_z_s_k, M_state, m_state, M_input_policy, m_input_policy):
    
    # GENERAL MPC MODEL BUILDER
    
    print(f"Start general MPC model contruction")
    start_time = time.perf_counter()

    MPC_model = gp.Model("MPC")

    Ns = centers.shape[0]

    # Optimization variables
    ub_u_matrix = np.zeros((PREDICTION_HORIZON,INPUTS_NUMBER))
    lb_u_matrix = np.zeros((PREDICTION_HORIZON,INPUTS_NUMBER))

    ub_x_matrix = np.zeros((PREDICTION_HORIZON+1,STATES_NUMBER))
    lb_x_matrix = np.zeros((PREDICTION_HORIZON+1,STATES_NUMBER))

    for i in range(PREDICTION_HORIZON):
        ub_u_matrix[i,:] = upper_bound_u 
        lb_u_matrix[i,:] = lower_bound_u

        ub_x_matrix[i,:] = upper_bound_x
        lb_x_matrix[i,:] = lower_bound_x

    ub_x_matrix[-1,:] = upper_bound_x
    lb_x_matrix[-1,:] = lower_bound_x

    u_tilde_k = MPC_model.addMVar((PREDICTION_HORIZON,INPUTS_NUMBER), ub=ub_u_matrix, lb=lb_u_matrix)
    x_tilde_k = MPC_model.addMVar((PREDICTION_HORIZON+1,STATES_NUMBER), ub=ub_x_matrix, lb=lb_x_matrix)


    # Optimization reconstruction variables 
    delta_s_k = MPC_model.addMVar((PREDICTION_HORIZON+1,Ns), vtype=GRB.BINARY)
    z_s_k = MPC_model.addMVar((PREDICTION_HORIZON+1,STATES_NUMBER,Ns), lb=lb_z_s_k, ub=ub_z_s_k)


    # State abstraction constraints for initial state

    EPSILON = 1e-6

    for cell_idx in range(Ns):
        MPC_model.addConstr(z_s_k[0,:,cell_idx] <= (M_state[cell_idx,:]-EPSILON)*delta_s_k[0,cell_idx])
        MPC_model.addConstr(z_s_k[0,:,cell_idx] >= (m_state[cell_idx,:]+EPSILON)*delta_s_k[0,cell_idx])

        MPC_model.addConstr(z_s_k[0,:,cell_idx] <= x_tilde_k[0,:] - lower_bound_x*(1 - delta_s_k[0,cell_idx]))
        MPC_model.addConstr(z_s_k[0,:,cell_idx] >= x_tilde_k[0,:] - upper_bound_x*(1 - delta_s_k[0,cell_idx]))
    # Cost definition
    cost = 0



    # Building prediction model
    for j in range(PREDICTION_HORIZON):

        # Input Constraint
        sum_upper_bounds = 0
        sum_lower_bounds = 0
        for cell_idx in range(Ns):
            sum_upper_bounds += M_input_policy[cell_idx,:]*delta_s_k[j,cell_idx]
            sum_lower_bounds += m_input_policy[cell_idx,:]*delta_s_k[j,cell_idx]

        
        MPC_model.addConstr(u_tilde_k[j,:] <= sum_upper_bounds)
        MPC_model.addConstr(u_tilde_k[j,:] >= sum_lower_bounds)

        # State abstraction constraints
        for cell_idx in range(Ns):
            MPC_model.addConstr(z_s_k[j+1,:,cell_idx] <= (M_state[cell_idx,:] - EPSILON)*delta_s_k[j+1,cell_idx])
            MPC_model.addConstr(z_s_k[j+1,:,cell_idx] >= (m_state[cell_idx,:] + EPSILON)*delta_s_k[j+1,cell_idx])

            MPC_model.addConstr(z_s_k[j+1,:,cell_idx] <= x_tilde_k[j+1,:] - lower_bound_x*(1 - delta_s_k[j+1,cell_idx]))
            MPC_model.addConstr(z_s_k[j+1,:,cell_idx] >= x_tilde_k[j+1,:] - upper_bound_x*(1 - delta_s_k[j+1,cell_idx]))

        # Abstraciton consistency (first true by assumption)
        delta_s_k_sum = 0

        for cell_idx in range(Ns):
            delta_s_k_sum += delta_s_k[j+1,cell_idx]

        MPC_model.addConstr(delta_s_k_sum <= 1)
        

        tau = 1.0

        # State transition matrix
        A  = np.array([[1, tau],
                        [0, 1]])
        
        # Input matrix
        B  = np.array([[tau**2/2],
                        [tau]])

        # Prediction step
        MPC_model.addConstr(x_tilde_k[j+1,:] == A @ x_tilde_k[j,:] + B @ u_tilde_k[j,:])

    # Optimization options
    MPC_model.Params.OutputFlag = 0             # print solver log (1 = on, 0 = off)
    MPC_model.Params.LogToConsole = 0
    MPC_model.Params.MIPFocus    = 3             # 1=feas, 2=bound, 3=optimality
    MPC_model.Params.MIPGap    = 0.05            # 5% gap 
    # MPC_model.Params.MIPGap    = 0.10            # 10% gap  
    # MPC_model.Params.MIPGap    = 0.001            # 0.1% gap  
    MPC_model.Params.Heuristics  = 0.25          # 0.1–0.5
    MPC_model.Params.Threads     = 0                    # 0 = all cores

    MPC_model.update()
    enlapsed_time = time.perf_counter() - start_time
    print(f"Model construction required: {enlapsed_time:.6f} seconds")

    # MPC MONTECARLO SIMULATION

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
        no_action = np.zeros(INPUTS_NUMBER)

        double_integrator_MPC = DoubleIntegratorDynamics(x0 + simulation_list_initial_state[i].copy())
        cost_policy_i = 0
        cost_policy_state_i = 0
        cost_policy_input_i = 0
        for k in range(SIMULATION_HORIZON+PREDICTION_HORIZON):
            print(f"MPC Experiment {i+1}/{NUMBER_EXPERIMENTS_MPC}")
            print(f"Time step {k}/{SIMULATION_HORIZON+PREDICTION_HORIZON}")
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
                
                x_trajectory[k+1,:] = double_integrator_MPC.step(u_trajectory[k,:],w_sequence[k])
                
                cost_policy_i += u_trajectory[k,:] @ R @ u_trajectory[k,:] + (target_cell_k - x_trajectory[k+1,:])@Q@(target_cell_k - x_trajectory[k+1,:])
                cost_policy_state_i += (target_cell_k - x_trajectory[k+1,:])@Q@(target_cell_k - x_trajectory[k+1,:])
                cost_policy_input_i += u_trajectory[k,:] @ R @ u_trajectory[k,:] 
                # cost_policy_i += u_trajectory[k,:] @ R @ u_trajectory[k,:] + (x_trajectory[k+1,:] - x_trajectory[k,:])@Q@(x_trajectory[k+1,:] - x_trajectory[k,:])
                MPC_model_k.dispose()
                simulation_steps_counter += 1
            cumulative_cost_MPC[i] = cost_policy_i
            cumulative_cost_MPC_state[i] = cost_policy_state_i
            cumulative_cost_MPC_input[i] = cost_policy_input_i

        # Store simulation      
        simulation_list_MPC_state.append(x_trajectory)
        simulation_list_MPC_input.append(u_trajectory)

        enlapsed_time_experiment = time.perf_counter() - start_time_experiment
        print(f"MPC simulation {i} required: {enlapsed_time_experiment:.6f} seconds")
        print(f"Total enlapsed time: {(time.perf_counter() - start_time):.6f} seconds")

        # TEMPORARY PLOT
        # for k in range(SIMULATION_HORIZON):
        #     ax.plot([x_trajectory[k,0],x_trajectory[k+1,0]], [x_trajectory[k,1],x_trajectory[k+1,1]], 'blue',linewidth=1)
        # if SHOW_PLOT == True:
        #     clear_output(wait=True)
        #     display(fig)

        # pbar_MPC.update(1)
    # pbar_MPC.close()

    enlapsed_time = time.perf_counter() - start_time
    print(f"MPC Montecarlo simulation time: {enlapsed_time:.6f} seconds")
    print(f"Average MPC simulation time: {(enlapsed_time/NUMBER_EXPERIMENTS_MPC):.6f} seconds")
    # empirical_satisfaction_probability = (NUMBER_EXPERIMENTS_MPC - violation_counter)/NUMBER_EXPERIMENTS_MPC

    # print(f"Empirical satisfaction probability MPC: {empirical_satisfaction_probability}")
    # Compute empirical satisfaction probability
    satisfaction_counter = 0
    for i in range(NUMBER_EXPERIMENTS_MPC):
        x_trajectory = simulation_list_MPC_state[i].copy()
        final_cell_index = get_cell_index(centers, x_trajectory[-1])
        if np.isin(final_cell_index,goal_centers_indices) == True:
            satisfaction_counter += 1
    empirical_satisfaction_probability_MPC = (satisfaction_counter)/NUMBER_EXPERIMENTS_MPC

    enlapsed_time = time.perf_counter() - start_time
    print(f"Policy Montecarlo simulation time: {enlapsed_time:.6f} seconds")
    print(f"Empirical satisfaction probability MPC: {empirical_satisfaction_probability_MPC}")

    return simulation_list_MPC_input, simulation_list_MPC_state, cumulative_cost_MPC, cumulative_cost_MPC_state, cumulative_cost_MPC_input, empirical_satisfaction_probability_MPC, enlapsed_time




# def mpc_montecarlo_simulation(NUMBER_EXPERIMENTS_MPC, SIMULATION_HORIZON, PREDICTION_HORIZON, STATES_NUMBER, INPUTS_NUMBER, simulation_list_noise, x0, simulation_list_initial_state, centers, goal_centers_indices, target_cell, policy, actions_inputs, R, Q, MPC_model, simulation_list_MPC_state, simulation_list_MPC_input, cumulative_cost_MPC, cumulative_cost_MPC_state, cumulative_cost_MPC_input):

#     start_time = time.perf_counter()
    
#     print(f"Solving MPC reconstruction problem")
    
#     Ns = centers.shape[0]

#     violation_counter = 0
    
    
#     backup_policy_usage_counter = 0
#     simulation_steps_counter = 0
#     for i in range(NUMBER_EXPERIMENTS_MPC):

        

#         start_time_experiment = time.perf_counter()
#         x_trajectory = np.zeros((SIMULATION_HORIZON+1+PREDICTION_HORIZON, STATES_NUMBER))
#         u_trajectory = np.zeros((SIMULATION_HORIZON+1+PREDICTION_HORIZON, INPUTS_NUMBER))
#         w_sequence = simulation_list_noise[i]

#         # x_trajectory[0,:] = x0
#         x_trajectory[0,:] = x0 + simulation_list_initial_state[i].copy()
#         no_action = np.array([0,0])

#         dubins_real_MPC = DubinsSmallDynamicsStochastic(x0 + simulation_list_initial_state[i].copy())
#         cost_policy_i = 0
#         cost_policy_state_i = 0
#         cost_policy_input_i = 0
#         for k in range(SIMULATION_HORIZON):
#             print(f"MPC Experiment {i+1}/{NUMBER_EXPERIMENTS_MPC}")
#             print(f"Time step {k}/{SIMULATION_HORIZON}")
#             x_k = x_trajectory[k,:]
#             initial_cell_idx = get_cell_index(centers, x_k)

#             action_index = policy[initial_cell_idx]

#             if (action_index == -1):
#                 u_trajectory[k,:] = no_action
#                 x_trajectory[k+1:SIMULATION_HORIZON+PREDICTION_HORIZON+1,:] = x_trajectory[k,:]
#                 break

#             else:

#                 # Run MPC

#                 # Create a local copy of the general MPC problem
#                 MPC_model_k = MPC_model.copy()

#                 # Set up initial condition constraints
#                 MPC_model_k.addConstr(x_tilde_k[0,:] == x_k)

#                 # Assign first boolean variable
#                 for cell_idx in range(0,Ns):
#                     if cell_idx == initial_cell_idx:
#                         MPC_model_k.addConstr(delta_s_k[0,cell_idx] == True)
#                     else: 
#                         MPC_model_k.addConstr(delta_s_k[0,cell_idx] == False)

#                 # Select target cell at time step k
#                 target_cell_k = target_cell[initial_cell_idx]
#                 # target_cell_k = target_cell_policy[initial_cell_idx]

#                 cost = 0
#                 # Construct cost function at time step k
#                 for j in range(PREDICTION_HORIZON):
#                     cost += u_tilde_k[j,:] @ R @ u_tilde_k[j,:] 
#                     cost += (target_cell_k - x_tilde_k[j+1,:]) @ Q @ (target_cell_k - x_tilde_k[j+1,:])
#                     # cost += (x_tilde_k[j+1,:] - x_tilde_k[j,:]) @ Q @ (x_tilde_k[j+1,:] - x_tilde_k[j,:])
                
#                 MPC_model_k.setObjective(cost, GRB.MINIMIZE)

#                 # Solve local MPC problem
#                 MPC_model_k.optimize()

#                 if MPC_model_k.Status == GRB.INFEASIBLE:
#                     print(f"INFEASIBLE PROBLEM")
#                     initial_cell_idx = get_cell_index(centers, x_trajectory[k,:])

#                     action_index = policy[initial_cell_idx]
#                     u_trajectory[k,:] = actions_inputs[action_index,:]
#                     backup_policy_usage_counter += 1
#                     print(f"Deploy backup policy")

#                 else:
#                     # Retrieving the control action
#                     opt_solution = MPC_model_k.getVars()[0:INPUTS_NUMBER]
#                     for j in range(INPUTS_NUMBER):
#                         u_trajectory[k,j] = opt_solution[j].X

#                 # Applying the control to the real system
                
#                 x_trajectory[k+1,:] = dubins_real_MPC.step(u_trajectory[k,:],w_sequence[k])
                
#                 cost_policy_state_i += (target_cell_k - x_trajectory[k+1,:])@Q@(target_cell_k - x_trajectory[k+1,:])
#                 cost_policy_input_i += u_trajectory[k,:] @ R @ u_trajectory[k,:] 
#                 cost_policy_i += (target_cell_k - x_trajectory[k+1,:])@Q@(target_cell_k - x_trajectory[k+1,:]) + u_trajectory[k,:] @ R @ u_trajectory[k,:] 

#                 MPC_model_k.dispose()
#                 simulation_steps_counter += 1
#             cumulative_cost_MPC[i] = cost_policy_i
#             cumulative_cost_MPC_state[i] = cost_policy_state_i
#             cumulative_cost_MPC_input[i] = cost_policy_input_i

#         # Store simulation      
#         simulation_list_MPC_state.append(x_trajectory)
#         simulation_list_MPC_input.append(u_trajectory)

#         enlapsed_time_experiment = time.perf_counter() - start_time_experiment
#         print(f"MPC simulation {i} required: {enlapsed_time_experiment:.6f} seconds")
#         print(f"Total enlapsed time: {(time.perf_counter() - start_time):.6f} seconds")


#     enlapsed_time = time.perf_counter() - start_time
#     print(f"MPC Montecarlo simulation time: {enlapsed_time:.6f} seconds")
#     print(f"Average MPC simulation time: {(enlapsed_time/NUMBER_EXPERIMENTS_MPC):.6f} seconds")
#     # Compute empirical satisfaction probability
#     satisfaction_counter = 0
#     for i in range(NUMBER_EXPERIMENTS_MPC):
#         x_trajectory = simulation_list_MPC_state[i].copy()
#         final_cell_index = get_cell_index(centers, x_trajectory[-1])
#         if np.isin(final_cell_index,goal_centers_indices) == True:
#             satisfaction_counter += 1
#     empirical_satisfaction_probability_MPC = (satisfaction_counter)/NUMBER_EXPERIMENTS_MPC

#     enlapsed_time = time.perf_counter() - start_time
#     print(f"Policy Montecarlo simulation time: {enlapsed_time:.6f} seconds")
#     print(f"Empirical satisfaction probability MPC: {empirical_satisfaction_probability_MPC}")


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