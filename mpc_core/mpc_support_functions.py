import numpy as np
import os
import pickle


def store_simulation_data(STORE_SIMULATION_DATA, model, NUMBER_PWA_REGIONS, lower_bound_x, upper_bound_x, NUMBER_EXPERIMENTS_MONTECARLO, NUMBER_EXPERIMENTS_MPC, expected_cost_policy, expected_cost_MPC, performance_improvement, expected_MPC_computation_time_step, backup_policy_usage_percentage, simulation_steps_counter, backup_policy_usage_counter, empirical_satisfaction_probability_MPC, empirical_satisfaction_probability_policy, simulation_list_MPC_input, simulation_list_MPC_state, simulation_list_policy_input, simulation_list_policy_state, simulation_list_noise, simulation_list_initial_state, Q, R, target_cell, M_state, m_state, Lp_balls, M_input, m_input, Ns, policy_offline, M_input_policy, m_input_policy,current_dir,SIMULATION_ID):
    # Store the simulation data in a .npy file
    if STORE_SIMULATION_DATA != "True":
        return
    
    if model == 'Double_integrator':
        mpc_simulation_folder = "mpc_simulation_double_integrator"
    elif model == 'Mountain_car':
        mpc_simulation_folder = "mpc_simulation_mountain_car"
    elif model == 'Dubins_small':
        mpc_simulation_folder = "mpc_simulation_small_dubin"

    simulation_data_MPC = {
                        # 'simulation_data': simulation_data,
                        'NUMBER_PWA_REGIONS': NUMBER_PWA_REGIONS,
                        'lower_bound_x': lower_bound_x,
                        'upper_bound_x': upper_bound_x,
                        'NUMBER_EXPERIMENTS_MONTECARLO': NUMBER_EXPERIMENTS_MONTECARLO,
                        'NUMBER_EXPERIMENTS_MPC': NUMBER_EXPERIMENTS_MPC,
                        'expected_cost_policy': expected_cost_policy,
                        'expected_cost_MPC': expected_cost_MPC,
                        'performance_improvement': performance_improvement,
                        'expected_MPC_computation_time_step': expected_MPC_computation_time_step,
                        'backup_policy_usage_percentage': backup_policy_usage_percentage,
                        'simulation_steps_counter': simulation_steps_counter,
                        'backup_policy_usage_counter': backup_policy_usage_counter,
                        'empirical_satisfaction_probability_MPC': empirical_satisfaction_probability_MPC,
                        'empirical_satisfaction_probability_policy': empirical_satisfaction_probability_policy,
                        'simulation_list_MPC_input': simulation_list_MPC_input,
                        'simulation_list_MPC_state':simulation_list_MPC_state, 
                        'simulation_list_policy_input': simulation_list_policy_input,
                        'simulation_list_policy_state': simulation_list_policy_state,
                        'simulation_list_noise': simulation_list_noise,
                        'simulation_list_initial_state': simulation_list_initial_state,
                        'Q': Q,
                        'R': R,
                        'target_cell': target_cell,
                        'M_state': M_state,
                        'm_state': m_state,
                        'Lp_balls': Lp_balls,
                        'M_input': M_input,
                        'm_input': m_input,
                        'Ns': Ns,
                        'policy_offline': policy_offline, 
                        'M_input_policy': M_input_policy,
                        'm_input_policy': m_input_policy,
                        # 'z_s': z_s,
                        # 'delta_s': delta_s
    }

    # Get the path of the required file
    file_path = os.path.join(current_dir, "mpc_simulation_data", mpc_simulation_folder, f"simulation_mpc_mountain_cars_{SIMULATION_ID}.pkl")

    # Load pickle file
    with open(file_path, "wb") as f:   
        pickle.dump(simulation_data_MPC, f)

def optimization_matrices_computation(upper_bounds, lower_bounds, actions_inputs, Lp_balls, centers, policy, PREDICTION_HORIZON, STATES_NUMBER, INPUTS_NUMBER, upper_bound_x, lower_bound_x):
    M_state = upper_bounds
    m_state = lower_bounds


    Ns = centers.shape[0]

    M_input = actions_inputs + Lp_balls
    m_input = actions_inputs - Lp_balls



    # Associated action
    policy_offline = np.zeros((Ns, INPUTS_NUMBER))

    # Associated bounds
    M_input_policy = np.zeros((Ns, INPUTS_NUMBER))
    m_input_policy = np.zeros((Ns, INPUTS_NUMBER))

    for cell_idx in range(0,Ns):

        no_action = np.zeros((INPUTS_NUMBER))
        action_idx = policy[cell_idx]
        
        if action_idx == -1:
            policy_offline[cell_idx,:] = no_action
            M_input_policy[cell_idx,:] = no_action
            m_input_policy[cell_idx,:] = no_action
        else:
            policy_offline[cell_idx,:] = actions_inputs[action_idx,:]
            M_input_policy[cell_idx,:] = actions_inputs[action_idx,:] + Lp_balls
            m_input_policy[cell_idx,:] = actions_inputs[action_idx,:] - Lp_balls

    ub_z_s_k = np.zeros((PREDICTION_HORIZON+1,STATES_NUMBER,Ns))
    lb_z_s_k = np.zeros((PREDICTION_HORIZON+1,STATES_NUMBER,Ns))

    for i in range (PREDICTION_HORIZON+1):
        for j in range(Ns):
            ub_z_s_k[i,:,j] = upper_bound_x
            lb_z_s_k[i,:,j] = lower_bound_x

    return M_state, m_state, M_input_policy, m_input_policy, ub_z_s_k, lb_z_s_k, policy_offline, M_input, m_input, Ns


# Support Functions

def reference_generator(centers, goal_centers, STATES_NUMBER):
    # Closest target cells computations
    # Number of partition cells 
    Ns = centers.shape[0]

    target_cell = np.zeros((Ns,STATES_NUMBER))

    for i in range(Ns):
        array_difference = goal_centers - centers[i,:]
        distances = np.zeros(len(goal_centers))
        for j in range(len(goal_centers)):
            distances[j] = np.linalg.norm(array_difference[j,:],1)
        min_idx = np.argmin(distances)
        target_cell[i,:] = goal_centers[min_idx,:].copy()

    return target_cell


def get_goal_centers(centers, goal_centers):
    goal_centers_indices = np.zeros(len(goal_centers))

    for i in range(len(goal_centers)):
        goal_centers_indices[i] = get_cell_index(centers,goal_centers[i])

    return goal_centers_indices.astype(int)


def get_cell_index(centers,state):

    distances = np.sum((centers - state)**2, axis=1)
    return np.argmin(distances)


def get_cell_distance(centers,state):

    distances = np.sum((centers - state)**2, axis=1)
    return np.sqrt(np.min(distances))

def noise_generator(NUMBER_EXPERIMENTS_MONTECARLO, SIMULATION_HORIZON, PREDICTION_HORIZON, STATES_NUMBER, mean, cov_noise, cov_initial_state):

    NUMBER_NOISE_INPUTS = cov_noise.shape[0]

    # std_dev_noise = pow(cov_noise,2) # standard deviation

    std_dev_initial_state = pow(cov_initial_state,2) # standard deviation

    simulation_list_noise = []
    simulation_list_initial_state = []

    np.random.seed(0) # for reproducibility
    for i in range(NUMBER_EXPERIMENTS_MONTECARLO):
        # Generate i.i.d. Gaussian noise
        noise_list = []
        for j in range(0, NUMBER_NOISE_INPUTS):
            std_dev_noise = pow(cov_noise[j],2)
            noise_list.append(np.random.normal(loc=mean, scale=std_dev_noise, size=SIMULATION_HORIZON+1+PREDICTION_HORIZON))
        w_sequence = np.array(noise_list).T
        x_initial = np.random.normal(loc=mean, scale=std_dev_initial_state, size=STATES_NUMBER)
        # print(w)
        simulation_list_noise.append(w_sequence)
        simulation_list_initial_state.append(x_initial)

    return simulation_list_noise, simulation_list_initial_state