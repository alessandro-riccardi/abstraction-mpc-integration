import numpy as np
from mpc_core.mpc_support_functions import get_cell_index


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