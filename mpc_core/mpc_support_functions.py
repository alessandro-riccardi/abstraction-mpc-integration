import numpy as np


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

    std_dev_noise = pow(cov_noise,2) # standard deviation

    std_dev_initial_state = pow(cov_initial_state,2) # standard deviation

    simulation_list_noise = []
    simulation_list_initial_state = []

    np.random.seed(0) # for reproducibility
    for i in range(NUMBER_EXPERIMENTS_MONTECARLO):
        # Generate i.i.d. Gaussian noise
        w_sequence = np.random.normal(loc=mean, scale=std_dev_noise, size=SIMULATION_HORIZON+1+PREDICTION_HORIZON)
        x_initial = np.random.normal(loc=mean, scale=std_dev_initial_state, size=STATES_NUMBER)
        # print(w)
        simulation_list_noise.append(w_sequence)
        simulation_list_initial_state.append(x_initial)

    return simulation_list_noise, simulation_list_initial_state