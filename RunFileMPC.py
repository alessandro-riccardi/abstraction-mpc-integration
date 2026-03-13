# %% MARK: Load libraries

import numpy as np
import os
import sys
import pickle
from mpc_core.Double_integrator_dynamics import * # make lower case, modify file
from mpc_core.mountain_car_dynamics import *
from mpc_core.Dubins_small_dynamics import * # make lower case, modify file
from mpc_core.options import parse_arguments_mpc
from mpc_core.mpc_support_functions import noise_generator
from mpc_core.mpc_support_functions import get_cell_index
from mpc_core.mpc_support_functions import get_cell_distance
from mpc_core.mpc_support_functions import get_goal_centers
from mpc_core.mpc_support_functions import reference_generator
from tqdm import tqdm
import gurobipy as gp
from gurobipy import GRB
import cvxpy as cp
import time

import sys

# sys.argv = ['RunFileMPC.py', 
#             '--model', 'Dubins_small', 
#             '--abstraction_data_nominal', 'abstraction_data_nopminal',  # abstraction_data_Dubins_small_04_01.pkl
#             '--abstraction_data', 'abstraction_data',
#             '--simulation_id', 'simulation_id',                         # "04_03_Final"
#             '--store_simulation_data', 'True',                          # abstraction_data_Dubins_small_04_04.pkl
#             '--plot_simulation', 'True'] 

sys.argv = ['RunFileMPC.py', 
            '--model', 'Double_integrator',  
            '--abstraction_data_nominal', 'abstraction_data_DoubleIntegrator_01',  # abstraction_data_Dubins_small_04_01.pkl
            '--abstraction_data', 'abstraction_data_DoubleIntegrator_02',
            '--simulation_id', 'simulation_id',                         # "04_03_Final"
            '--store_simulation_data', 'True',                          # abstraction_data_Dubins_small_04_04.pkl
            '--plot_simulation', 'True',
            '--simulation_horizon', '25',
            '--prediction_horizon', '3',
            '--number_experiments', '3',
            '--mean_noise', '0.0',
            '--cov_noise', '0.1',
            '--cov_initial_state', '0.25'] 

if __name__ == '__main__':
    
    # %% MARK: Load abstraciton data

    # parse arguments
    args = parse_arguments_mpc()
    
    model = args.model

    if model == 'Double_integrator':
        abstraction_folder = "abstraction_data_double_integrator"
    elif model == 'Mountain_car':
        abstraction_folder = "abstraction_data_mountain_car"
    elif model == 'Dubins_small':
        abstraction_folder = "abstraction_data_small_dubin"

    abstraction_data_nominal = args.abstraction_data_nominal
    abstraction_data = args.abstraction_data

    # get current directory
    current_dir = os.getcwd()

    # locate abstraction data
    file_path_nominal = os.path.join(current_dir, "abstraction_data", abstraction_folder, f"{abstraction_data_nominal}.pkl")
    file_path = os.path.join(current_dir, "abstraction_data", abstraction_folder, f"{abstraction_data}.pkl")

    # Load nominal abstraction data
    with open(file_path_nominal, "rb") as f:   
        simulation_data_nominal = pickle.load(f)

    # Load abstraction data
    with open(file_path, "rb") as f:   
        simulation_data = pickle.load(f)

    # define nominal policy
    policy_inputs_nominal = simulation_data_nominal['policy_inputs']
    actions_inputs_nominal = simulation_data_nominal['actions.inputs']
    policy_nominal = simulation_data_nominal['policy']

    # Import abstraction data
    policy_inputs = simulation_data['policy_inputs']
    actions_inputs = simulation_data['actions.inputs']
    upper_bounds = simulation_data['upper_bounds']
    lower_bounds = simulation_data['lower_bounds']
    all_vertices = simulation_data['all_vertices']
    centers = simulation_data['centers']
    goal_centers = simulation_data['goal_centers']
    critical_centers = simulation_data['critical_centers']
    critical_centers_indexes = simulation_data['critical_centers_indexes']
    policy = simulation_data['policy']
    cell_width = simulation_data['cell_width']
    Lp_balls = simulation_data['epsilons']


# %% Simulation parameters

    SIMULATION_HORIZON = args.simulation_horizon
    PREDICTION_HORIZON = args.prediction_horizon

    NUMBER_EXPERIMENTS_MONTECARLO = args.number_experiments
    NUMBER_EXPERIMENTS_MPC = args.number_experiments


# %% Model parameters

    STATES_NUMBER = simulation_data['STATES_NUMBER']
    INPUTS_NUMBER = simulation_data['INPUTS_NUMBER']
    initial_state = simulation_data['initial_state']
    
# %% MARK: Noise sequence generation    

    mean = args.mean_noise
    cov_noise = args.cov_noise
    cov_initial_state = args.cov_initial_state

    simulation_list_noise, simulation_list_initial_state = noise_generator(NUMBER_EXPERIMENTS_MONTECARLO, SIMULATION_HORIZON, PREDICTION_HORIZON, STATES_NUMBER, mean=0.0, cov_noise=0.1, cov_initial_state=0.25)
    
# %% MARK: Reference generator

    goal_centers_indices = get_goal_centers(centers, goal_centers)

    target_cell = reference_generator(centers, goal_centers, STATES_NUMBER)