import argparse
import json


def parse_arguments_mpc():
    '''
    Function to parse arguments provided

    :return: Object with all arguments
    '''

    # Options
    parser = argparse.ArgumentParser(prefix_chars='--')
    # parser.add_argument('--debug', action=argparse.BooleanOptionalAction, default=False,
    #                     help="If True, perform additional checks to debug python")
    # parser.add_argument('--seed', type=int, default=0,
    #                     help="Seed for random number generators (Jax, Numpy)")
    # parser.add_argument('--decimals', type=int, default=4,
    #                     help="Number of decimals to work with for storing probabilities")
    

    # parser.add_argument('--gpu', action=argparse.BooleanOptionalAction, default=False,
    #                     help="If true, run on GPU. Otherwise, run on CPU")

    parser.add_argument('--model', type=str, default='Drone2D',
                        help="Benchmark model to run")

    parser.add_argument('--abstraction_data', type=str, default='abstraction_data',
                        help="File loading abstracted system to use in the MPC")
    
    parser.add_argument('--abstraction_data_nominal', type=str, default='abstraction_data_nominal',
                        help="File loading nominal abstracted system to use in the MPC (optional; if not provided, the same abstraction will be used for both nominal and robust MPC)")
    
    parser.add_argument('--simulation_id', type=str, default='simulation_id',
                        help="ID of the simulation used storing results")
    
    parser.add_argument('--store_simulation_data', type=str, default='False',
                        help="If True, store data from the MPC simulation (state, action, etc.)")
    
    parser.add_argument('--plot_simulation', type=str, default='False',
                        help="If True, plot the results of the MPC simulation")
    
    parser.add_argument('--simulation_horizon', type=int, default=25,
                        help="Horizon of the control simulation")
    
    parser.add_argument('--prediction_horizon', type=int, default=3, 
                        help="Prediction horizon of the MPC controller")
    
    parser.add_argument('--number_experiments', type=int, default=10,
                        help="Number of experiments to run for the Monte Carlo simulation and the MPC simulation")  
    
    parser.add_argument('--mean_noise', type=float, default=0.0,
                        help="Mean of the noise affecting the system")
    
    parser.add_argument('--cov_noise', type=json.loads,
                        help="Covariance of the noise affecting the system")
    
    parser.add_argument('--cov_initial_state', type=float, default=0.25,
                        help="Covariance of the initial state of the system")
    
    parser.add_argument('--input_weight', type=json.loads,
                        help="Weight matrix for the input in the MPC cost function (as a string that can be converted to a list)")  
    
    parser.add_argument('--state_weight', type=json.loads,
                        help="Weight matrix for the state in the MPC cost function (as a string that can be converted to a list)")

    parser.add_argument('--number_pwa_regions', type=int, default=100,
                        help="Number of PWA regions to use for the hybrid approximation of the nonlinear dynamics")

    # parser.add_argument('--model_version', type=int, default=0,
    #                     help="Version of the model to use (optinal; 0 by default)")
    # parser.add_argument('--checker', type=str, default='storm',
    #                     help="Model checker to use (prism or storm)")
    # parser.add_argument('--prism_dir', type=str, default='~/Documents/Tools/prism/prism/bin/prism',
    #                     help="Directory where Prism is located")

    # parser.add_argument('--mode', type=str, default='fori_loop',
    #                     help="Should be one of 'fori_loop', 'vmap', 'python'")
    # parser.add_argument('--batch_size', type=int, default=1_000_000,
    #                     help="Batch size for functions vectorized with Jax")

    # # Plotting options
    # parser.add_argument('--plot_grid', action=argparse.BooleanOptionalAction, default=False,
    #                     help="If True, plot unit grids in figures")
    # parser.add_argument('--plot_title', action=argparse.BooleanOptionalAction, default=False,
    #                     help="If True, plot titles in figures")
    # parser.add_argument('--plot_ticks', action=argparse.BooleanOptionalAction, default=False,
    #                     help="If True, plot ticks in figures")

    # Parse arguments
    args = parser.parse_args()

    return args
