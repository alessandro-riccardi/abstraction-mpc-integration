# %% MARK: Load Libraries

import numpy as np
import os
import sys
import pickle
from mpc_core.double_integrator_dynamics import *
from mpc_core.mountain_car_dynamics import *
from mpc_core.dubins_small_dynamics import *
from mpc_core.options import parse_arguments
from tqdm import tqdm
import gurobipy as gp
from gurobipy import GRB
import cvxpy as cp
import time

import sys

sys.argv = ['RunFileMPC.py', 
            '--model', 'Dubins_small', 
            '--abstraction_data', 'abstraction_data', 
            '--simulation_id', 'simulation_id', 
            '--store_simulation_data', True, 
            '--plot_simulation', True] 


if __name__ == '__main__':
    
    args = parse_arguments_mpc()



    a = 1