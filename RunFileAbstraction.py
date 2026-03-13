'''
This is the main Python file for DynAbs-JAX.
The file can be run from the terminal as

```Python3 RunFile.py --model <model-name> ...```

For all available arguments, please see the function :func:`core.options.parse_arguments`.
'''

# %load_ext autoreload
# %autoreload 2

import datetime
import os
import time
from pathlib import Path
from core.simulate import MonteCarloSim_passive
from plotting.traces import plot_traces
import stormpy

import jax
import numpy as np

import benchmarks
from core.Gaussian_probabilities import compute_probability_intervals
from core.actions_forward import RectangularForward
from core.model import parse_linear_model, parse_nonlinear_model
from core.options import parse_arguments
from core.partition import RectangularPartition

# from core.utils import store_abstraction_data

# To store indivdual variables
import pickle


# To run code in notebook
import sys
model_version = 0

# Dubins Car
# sys.argv = ['RunFileAbstraction.py', '--model', 'Dubins', '--model_version', f'{model_version}']

# Small Dubins Car
# sys.argv = ['RunFileAbstraction.py', '--model', 'Dubins_small', '--epsilons', '[0.00, 0.00]', '--store_abstraction_data', "True", '--simulation_id', '01', '--batch_size', '3000']
# sys.argv = ['RunFileAbstraction.py', '--model', 'Dubins_small', '--epsilons', '[0.10, 0.20]', '--store_abstraction_data', "True", '--simulation_id', '02', '--batch_size', '3000']
# sys.argv = ['RunFileAbstraction.py', '--model', 'Dubins_small', '--epsilons', '[0.15, 0.30]', '--store_abstraction_data', "True", '--simulation_id', '03', '--batch_size', '3000']
# sys.argv = ['RunFileAbstraction.py', '--model', 'Dubins_small', '--epsilons', '[0.20, 0.40]', '--store_abstraction_data', "True", '--simulation_id', '04', '--batch_size', '3000']
# sys.argv = ['RunFileAbstraction.py', '--model', 'Dubins_small', '--epsilons', '[0.16, 0.32]', '--store_abstraction_data', "True", '--simulation_id', '05', '--batch_size', '3000']
# sys.argv = ['RunFileAbstraction.py', '--model', 'Dubins_small', '--epsilons', '[0.17, 0.34]', '--store_abstraction_data', "True", '--simulation_id', '06', '--batch_size', '3000']
# sys.argv = ['RunFileAbstraction.py', '--model', 'Dubins_small', '--epsilons', '[0.18, 0.36]', '--store_abstraction_data', "True", '--simulation_id', '07', '--batch_size', '3000']


# Pendulum
# sys.argv = ['RunFileAbstraction.py', '--model', 'Pendulum', '--epsilons', 'epsilons', '--store_abstraction_data', "True", '--simulation_id', 'simulation_id','--batch_size', '30000']

# Mountain Car
# sys.argv = ['RunFileAbstraction.py', '--model', 'MountainCar', '--epsilons', '0.00', '--store_abstraction_data', "True", '--simulation_id', '01','--batch_size', '30000']
# sys.argv = ['RunFileAbstraction.py', '--model', 'MountainCar', '--epsilons', '0.10', '--store_abstraction_data', "True", '--simulation_id', '02','--batch_size', '30000']
sys.argv = ['RunFileAbstraction.py', '--model', 'MountainCar', '--epsilons', '0.20', '--store_abstraction_data', "True", '--simulation_id', '03','--batch_size', '30000']
# sys.argv = ['RunFileAbstraction.py', '--model', 'MountainCar', '--epsilons', '0.15', '--store_abstraction_data', "True", '--simulation_id', '04','--batch_size', '30000']

# Double Integrator
# sys.argv = ['RunFileAbstraction.py', '--model', 'DoubleIntegrator', '--epsilons', '0.0', '--store_abstraction_data', "True", '--simulation_id', '01','--batch_size', '30000']
# sys.argv = ['RunFileAbstraction.py', '--model', 'DoubleIntegrator', '--epsilons', '0.1', '--store_abstraction_data', "True", '--simulation_id', '02','--batch_size', '30000']
# sys.argv = ['RunFileAbstraction.py', '--model', 'DoubleIntegrator', '--epsilons', '0.2', '--store_abstraction_data', "True", '--simulation_id', '03','--batch_size', '30000']
# sys.argv = ['RunFileAbstraction.py', '--model', 'DoubleIntegrator', '--epsilons', '0.3', '--store_abstraction_data', "True", '--simulation_id', '04','--batch_size', '30000']
# sys.argv = ['RunFileAbstraction.py', '--model', 'DoubleIntegrator', '--epsilons', '0.4', '--store_abstraction_data', "True", '--simulation_id', '05','--batch_size', '30000']
# sys.argv = ['RunFileAbstraction.py', '--model', 'DoubleIntegrator', '--epsilons', '0.5', '--store_abstraction_data', "True", '--simulation_id', '06','--batch_size', '30000']
# sys.argv = ['RunFileAbstraction.py', '--model', 'DoubleIntegrator', '--epsilons', '1.0', '--store_abstraction_data', "True", '--simulation_id', '07','--batch_size', '30000']
# sys.argv = ['RunFileAbstraction.py', '--model', 'DoubleIntegrator', '--epsilons', '2.0', '--store_abstraction_data', "True", '--simulation_id', '08','--batch_size', '30000']

# Triple Integrator
# sys.argv = ['RunFileAbstraction.py', '--model', 'TripleIntegrator', '--epsilons', 'epsilons', '--store_abstraction_data', True, '--simulation_id', 'simulation_id','--batch_size', '30000']

# %run RunFile.py

if __name__ == '__main__':
    jax.config.update("jax_default_matmul_precision", "high")

    args = parse_arguments()
    if args.gpu:
        jax.config.update('jax_platform_name', 'gpu')
    else:
        jax.config.update('jax_platform_name', 'cpu')

    simulation_id = args.simulation_id

    print('=== JAX STATUS ===')
    print(f'Devices available: {jax.devices()}')
    from jax.lib import xla_bridge

    print(f'Jax runs on: {xla_bridge.get_backend().platform}')
    print('==================\n')

    np.random.seed(args.seed)
    args.jax_key = jax.random.PRNGKey(args.seed)

    # In debug mode, configure jax to use Float64 (for more accurate computations)
    if args.debug:
        from jax import config

        config.update("jax_enable_x64", True)

    # Set current working directory
    args.cwd = os.path.dirname(os.path.abspath(__file__))
    args.root_dir = Path(args.cwd)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f'Run started at {stamp} using arguments:')
    for key, val in vars(args).items():
        print(' - `' + str(key) + '`: ' + str(val))
    print('\n==============================\n')

    # Define and parse model
    if args.model == 'Dubins':
        base_model = benchmarks.Dubins(args)
    elif args.model == 'Dubins_small':
        base_model = benchmarks.Dubins_small(args)
    elif args.model == 'Drone4D':
        base_model = benchmarks.Drone4D(args)
    elif args.model == 'Pendulum':
        base_model = benchmarks.Pendulum(args)
    elif args.model == 'MountainCar':
        base_model = benchmarks.MountainCar(args)
    elif args.model == 'Vanderpol':
        base_model = benchmarks.Vanderpol(args)
    elif args.model == 'DoubleIntegrator':
        base_model = benchmarks.DoubleIntegrator(args)
    elif args.model == 'TripleIntegrator':
        base_model = benchmarks.TripleIntegrator(args)
    else:
        assert False, f"The passed model '{args.model}' could not be found"

    t = time.time()

    # Parse given model
    if base_model.linear:
        model = parse_linear_model(base_model)
    else:
        model = parse_nonlinear_model(base_model)

    # Create partition of the continuous state space into convex polytope
    partition = RectangularPartition(model=model)
    print(f"(Number of states: {len(partition.regions['idxs'])})\n")
    
    # Create actions based on forward reachable sets
    actions = RectangularForward(partition=partition, model=model, debug=False)

    # With forward reachability, every action is enabled in every state
    enabled_actions = np.full((len(partition.regions['centers']), len(actions.idxs)), fill_value=True, dtype=np.bool)

    print(f"(Number of actions in each state: {np.sum(np.any(enabled_actions, axis=0))})\n")

    P_full, P_id, P_absorbing = compute_probability_intervals(args, model, partition, actions.frs, actions.max_slice)

    # %% Model checking

    from core.imdp import BuilderStorm

    # Compute optimal policy on the iMDP abstraction
    print('\nCreate iMDP using storm...')

    P_full[0]

    # Build interval MDP via StormPy
    builderS = BuilderStorm(partition=partition,
                            actions=actions,
                            states=np.array(partition.regions['idxs']),
                            x0=model.x0,
                            goal_regions=np.array(partition.goal['idxs']),
                            critical_regions=np.array(partition.critical['idxs']),
                            P_full=P_full,
                            P_id=P_id,
                            P_absorbing=P_absorbing)

    print(f'- Generating abstraction took: {(time.time() - t):.3f} sec.')
    print(builderS.imdp)


    t = time.time()
    result = builderS.compute_reach_avoid() 
    policy, policy_inputs = builderS.get_policy(actions)
    print(f'- Verify with storm took: {(time.time() - t):.3f} sec.')
    print('Total sum of reach probs:', np.sum(builderS.results))
    print('In state {}: {}'.format(model.x0, builderS.get_value_from_tuple(model.x0, partition)))

    # %% Simulations, plots, and storage

    from core.simulate import MonteCarloSim
    from plotting.traces import plot_traces
    from plotting.heatmap import heatmap

    sim = MonteCarloSim(model, partition, policy, policy_inputs, model.x0, verbose=False, iterations=10)
    print('Empirical satisfaction probability:', sim.results['satprob'])


    # Store data

    upper_bounds = np.array(partition.regions['upper_bounds'])
    lower_bounds = np.array(partition.regions['lower_bounds'])
    all_vertices = np.array(partition.regions['all_vertices'])
    centers = np.array(partition.regions['centers'])


    goal_centers = np.array(partition.goal['centers'])
    critical_centers = np.array(partition.critical['centers'])
    critical_centers_indexes = np.array(partition.critical['idxs'])

    cell_width = np.array(partition.cell_width)

    abstraction_data = {
                    # 'partition': partition, # loading this requires jax
                    # 'regions': regions, # loading this requires jax
                    # 'actions': actions, # loading this requires jax
                    'policy_inputs': policy_inputs,
                    'actions.inputs': actions.inputs,
                    'upper_bounds': upper_bounds,
                    'lower_bounds': lower_bounds,
                    'all_vertices': all_vertices,
                    'centers': centers,
                    'goal_centers': goal_centers,
                    'critical_centers': critical_centers,
                    'critical_centers_indexes': critical_centers_indexes,
                    'policy': policy,
                    'cell_width': cell_width,
                    'initial_state': model.x0,
                    'epsilons': model.epsilons,
                    'STATES_NUMBER': model.n,
                    'INPUTS_NUMBER': model.p

    }

    # Where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    data_folder = ""

    if args.model == 'Pendulum':
        data_folder = "abstraction_data_pendulum"
    if args.model == 'Dubins_small':
        data_folder = "abstraction_data_small_dubin"
    if args.model == 'MountainCar':
        data_folder = "abstraction_data_mountain_car"
    if args.model == 'DoubleIntegrator':
        data_folder = "abstraction_data_double_integrator"

    # To store results

    if args.store_abstraction_data == "True":
        # Get the path of the required file
        file_path = os.path.join(current_dir, "abstraction_data", data_folder, f"abstraction_data_{args.model}_{simulation_id}.pkl")

        # Load pickle file
        with open(file_path, "wb") as f:   
            pickle.dump(abstraction_data, f)

    #  Plots
    plot_traces(args, stamp, model.plot_dimensions, partition, model, sim.results['traces'], line=False, num_traces=10, add_unsafe_box=False,)
    heatmap(args, stamp, idx_show=model.plot_dimensions, slice_values=np.zeros(model.n), partition=partition, results=builderS.results, filename="heatmap_satprob")
    
    # %% To save individual abstraction data (occupies a lot of space, so not used by default)

    # if args.store_abstraction_data == True:
    #     store_abstraction_data(simulation_id, args, stamp, key, val, base_model, t, model, partition, actions, enabled_actions, P_full, P_id, P_absorbing, result, policy, policy_inputs, sim)
     

