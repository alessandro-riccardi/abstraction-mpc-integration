import numpy as np
from mpc_core.mpc_support_functions import get_cell_index


def mpc_montecarlo_simulation_performance_evaluation(NUMBER_EXPERIMENTS_MPC, NUMBER_EXPERIMENTS_MONTECARLO, simulation_list_policy_state, simulation_list_MPC_state, empirical_satisfaction_probability_MPC, centers, goal_centers_indices, SIMULATION_HORIZON, cumulative_cost_policy, cumulative_cost_policy_state, cumulative_cost_policy_input, cumulative_cost_MPC, cumulative_cost_MPC_state, cumulative_cost_MPC_input, enlapsed_time, simulation_steps_counter, backup_policy_usage_counter):
    expected_cost_policy = 0
    expected_cost_policy_state = 0
    expected_cost_policy_input = 0
    satisfaction_counter = 0
    for i in range(NUMBER_EXPERIMENTS_MONTECARLO):
        x_trajectory = simulation_list_policy_state[i].copy()
        final_cell_index = get_cell_index(centers, x_trajectory[-1])
        if np.isin(final_cell_index,goal_centers_indices) == True:
            expected_cost_policy += cumulative_cost_policy[i]
            expected_cost_policy_state += cumulative_cost_policy_state[i]
            expected_cost_policy_input += cumulative_cost_policy_input[i]
            satisfaction_counter += 1
    expected_cost_policy = expected_cost_policy/satisfaction_counter
    expected_cost_policy_state = expected_cost_policy_state/satisfaction_counter
    expected_cost_policy_input = expected_cost_policy_input/satisfaction_counter


    expected_cost_MPC = 0
    expected_cost_MPC_state = 0
    expected_cost_MPC_input = 0
    satisfaction_counter = 0
    for i in range(NUMBER_EXPERIMENTS_MONTECARLO):
        x_trajectory = simulation_list_MPC_state[i].copy()
        final_cell_index = get_cell_index(centers, x_trajectory[-1])
        if np.isin(final_cell_index,goal_centers_indices) == True:
            expected_cost_MPC += cumulative_cost_MPC[i]
            expected_cost_MPC_state += cumulative_cost_MPC_state[i]
            expected_cost_MPC_input += cumulative_cost_MPC_input[i]
            satisfaction_counter += 1
    expected_cost_MPC = expected_cost_MPC/satisfaction_counter
    expected_cost_MPC_state = expected_cost_MPC_state/satisfaction_counter
    expected_cost_MPC_input = expected_cost_MPC_input/satisfaction_counter

    performance_improvement = 1-(expected_cost_MPC/expected_cost_policy)
    performance_improvement_state = 1-(expected_cost_MPC_state/expected_cost_policy_state)
    performance_improvement_input = 1-(expected_cost_MPC_input/expected_cost_policy_input)
    expected_MPC_computation_time_step = enlapsed_time/(simulation_steps_counter - backup_policy_usage_counter)
    backup_policy_usage_percentage = backup_policy_usage_counter/simulation_steps_counter


    #  Path metrics

    total_cost_policy = 0
    satisfaction_counter = 0
    for i in range(NUMBER_EXPERIMENTS_MONTECARLO):
        x_trajectory = simulation_list_policy_state[i].copy()
        final_cell_index = get_cell_index(centers, x_trajectory[-1])
        if np.isin(final_cell_index,goal_centers_indices) == True:
            x_trajectory_policy = simulation_list_policy_state[i]
            satisfaction_counter +=1
            for j in range(SIMULATION_HORIZON):
                total_cost_policy += np.linalg.norm(x_trajectory_policy[j+1,:] - x_trajectory_policy[j,:])
    total_cost_policy = total_cost_policy/satisfaction_counter


    total_cost_MPC = 0
    satisfaction_counter = 0
    for i in range(NUMBER_EXPERIMENTS_MONTECARLO):
        x_trajectory = simulation_list_MPC_state[i].copy()
        final_cell_index = get_cell_index(centers, x_trajectory[-1])
        if np.isin(final_cell_index,goal_centers_indices) == True:
            x_trajectory_MPC = simulation_list_MPC_state[i]
            satisfaction_counter +=1
            for j in range(SIMULATION_HORIZON):
                total_cost_MPC += np.linalg.norm(x_trajectory_MPC[j+1,:] - x_trajectory_MPC[j,:])

    total_cost_MPC = total_cost_MPC/satisfaction_counter

    # Print results

    print(f"MPC Montecarlo simulation time: {enlapsed_time:.6f} seconds")
    print(f"Average MPC simulation time: {(enlapsed_time/NUMBER_EXPERIMENTS_MPC):.6f} seconds")
    print(f"Expected MPC step computation time: {expected_MPC_computation_time_step}")

    print(f"Expected cost associated to the policy: {expected_cost_policy}")
    print(f"Expected cost associated to the policy state: {expected_cost_policy_state}")
    print(f"Expected cost associated to the policy input: {expected_cost_policy_input}")

    print(f"Expected cost associated to the MPC: {expected_cost_MPC}")
    print(f"Expected cost associated to the MPC state: {expected_cost_MPC_state}")
    print(f"Expected cost associated to the MPC input: {expected_cost_MPC_input}")

    print(f"MPC performance improvement w.r.t. policy: {performance_improvement}")
    print(f"MPC performance improvement w.r.t. policy state: {performance_improvement_state}")
    print(f"MPC performance improvement w.r.t. policy input: {performance_improvement_input}")


    print(f"Expected path length policy: {total_cost_policy}")
    print(f"Expected path length MPC: {total_cost_MPC}")
    print(f"MPC performance improvement in trajectory length w.r.t. policy: {1-(total_cost_MPC/total_cost_policy)}")

    print(f"Empirical satisfaction probability MPC: {empirical_satisfaction_probability_MPC}")
    print(f"Percentage of backup policy usage: {backup_policy_usage_percentage}")

    return expected_cost_MPC, expected_cost_MPC_state, expected_cost_MPC_input, performance_improvement, performance_improvement_state, performance_improvement_input, total_cost_policy, total_cost_MPC, backup_policy_usage_percentage, expected_MPC_computation_time_step

def policy_montecarlo_simulation_performance_evaluation(NUMBER_EXPERIMENTS_MONTECARLO, simulation_list_policy_state, centers, goal_centers_indices, SIMULATION_HORIZON, cumulative_cost_policy, cumulative_cost_policy_state, cumulative_cost_policy_input):
    expected_cost_policy = 0
    expected_cost_policy_state = 0
    expected_cost_policy_input = 0
    satisfaction_counter = 0
    for i in range(NUMBER_EXPERIMENTS_MONTECARLO):
        x_trajectory = simulation_list_policy_state[i].copy()
        final_cell_index = get_cell_index(centers, x_trajectory[-1])
        if np.isin(final_cell_index,goal_centers_indices) == True:
            expected_cost_policy += cumulative_cost_policy[i]
            expected_cost_policy_state += cumulative_cost_policy_state[i]
            expected_cost_policy_input += cumulative_cost_policy_input[i]
            satisfaction_counter += 1
    if satisfaction_counter > 0:
        expected_cost_policy = expected_cost_policy/satisfaction_counter
        expected_cost_policy_state = expected_cost_policy_state/satisfaction_counter
        expected_cost_policy_input = expected_cost_policy_input/satisfaction_counter

    total_cost_policy = 0
    satisfaction_counter = 0
    for i in range(NUMBER_EXPERIMENTS_MONTECARLO):
        x_trajectory = simulation_list_policy_state[i].copy()
        final_cell_index = get_cell_index(centers, x_trajectory[-1])
        if np.isin(final_cell_index,goal_centers_indices) == True:
            x_trajectory_policy = simulation_list_policy_state[i]
            satisfaction_counter +=1
            for j in range(SIMULATION_HORIZON):
                total_cost_policy += np.linalg.norm(x_trajectory_policy[j+1,:] - x_trajectory_policy[j,:])
    if satisfaction_counter > 0:
        total_cost_policy = total_cost_policy/satisfaction_counter
    print(f"Expected cost associated to the policy: {expected_cost_policy}")
    print(f"Expected cost associated to the policy state: {expected_cost_policy_state}")
    print(f"Expected cost associated to the policy input: {expected_cost_policy_input}")
    print(f"Expected path length policy: {total_cost_policy}")

    return expected_cost_policy, expected_cost_policy_state, expected_cost_policy_input, total_cost_policy 