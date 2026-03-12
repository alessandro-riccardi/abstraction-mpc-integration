import numpy as np
import pickle
import os


def create_batches(data_length, batch_size):
    '''
    Create batches for the given data and batch size. Returns the start and end indices to iterate over.

    :param data_length: Total number of data points.
    :param batch_size: Number of points per batch.
    :return: Each batch is represented by the slice [starts[i]:ends[i]].
    '''

    num_batches = np.ceil(data_length / batch_size).astype(int)
    starts = (np.arange(num_batches) * batch_size).astype(int)
    ends = (np.minimum(starts + batch_size, data_length)).astype(int)

    return starts, ends


def lexsort4d(array):
    idxs = np.lexsort((
        array[:, 3],
        array[:, 2],
        array[:, 1],
        array[:, 0]
    ))

    return array[idxs]


def cm2inch(*tupl):
    '''
    Convert centimeters to inches
    '''

    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def remove_consecutive_duplicates(trace):
    '''
    Remove consecutive duplicates from a given trace.

    :param trace:
    :return: Trace without duplicates
    '''
    done = False
    i = 0
    while not done:
        # If same as next entry, remove it
        if i >= len(trace) - 1:
            done = True
        else:
            if np.all(trace[i] == trace[i + 1]):
                trace = trace[i + 1:]
            else:
                i += 1

    return trace

def store_abstraction_data(simulation_id, args, stamp, key, val, base_model, t, model, partition, actions, enabled_actions, P_full, P_id, P_absorbing, result, policy, policy_inputs, sim):
    '''
    Store abstraction data in a pickle file.

    :param data: Data to store
    :param filename: Name of the file to store the data in
    '''

    output_dir = 'abstraction_data'
    os.makedirs(output_dir, exist_ok=True)

    output_dir = 'abstraction_data'
    os.makedirs(output_dir, exist_ok=True)


    # args
    filename = f'simulation_{simulation_id}_{args.model}_args.pkl'
    filepath = os.path.join(output_dir, filename)
    pickle.dump(args, open(filepath, 'wb')) 

    # stamp
    filename = f'simulation_{simulation_id}_{args.model}_stamp.pkl'
    filepath = os.path.join(output_dir, filename)
    pickle.dump(stamp, open(filepath, 'wb')) 

    # key
    filename = f'simulation_{simulation_id}_{args.model}_key.pkl'
    filepath = os.path.join(output_dir, filename)
    pickle.dump(key, open(filepath, 'wb'))

    # val
    filename = f'simulation_{simulation_id}_{args.model}_val.pkl'
    filepath = os.path.join(output_dir, filename)
    pickle.dump(val, open(filepath, 'wb'))  

    # base_model
    filename = f'simulation_{simulation_id}_{args.model}_base_model.pkl'
    filepath = os.path.join(output_dir, filename)
    pickle.dump(base_model, open(filepath, 'wb')) 

    # t
    filename = f'simulation_{simulation_id}_{args.model}_t.pkl'
    filepath = os.path.join(output_dir, filename)
    pickle.dump(t, open(filepath, 'wb')) 

    # model
    filename = f'simulation_{simulation_id}_{args.model}_model.pkl'
    filepath = os.path.join(output_dir, filename)
    pickle.dump(model, open(filepath, 'wb'))
    
    # partition
    filename = f'simulation_{simulation_id}_{args.model}_partition.pkl'
    filepath = os.path.join(output_dir, filename)
    pickle.dump(partition, open(filepath, 'wb'))

    # actions
    filename = f'simulation_{simulation_id}_{args.model}_actions.pkl'
    filepath = os.path.join(output_dir, filename)
    pickle.dump(actions, open(filepath, 'wb'))

    # enabled_actions
    filename = f'simulation_{simulation_id}_{args.model}_enabled_actions.pkl'
    filepath = os.path.join(output_dir, filename)
    pickle.dump(enabled_actions, open(filepath, 'wb'))

    # P_full
    filename = f'simulation_{simulation_id}_{args.model}_P_full.pkl'
    filepath = os.path.join(output_dir, filename)
    pickle.dump(P_full, open(filepath, 'wb'))

    # P_id
    filename = f'simulation_{simulation_id}_{args.model}_P_id.pkl'
    filepath = os.path.join(output_dir, filename)
    pickle.dump(P_id, open(filepath, 'wb'))

    # P_absorbing
    filename = f'simulation_{simulation_id}_{args.model}_P_absorbing.pkl'
    filepath = os.path.join(output_dir, filename)
    pickle.dump(P_absorbing, open(filepath, 'wb'))

    # result
    filename = f'simulation_{simulation_id}_{args.model}_result.pkl'
    filepath = os.path.join(output_dir, filename)
    pickle.dump(result, open(filepath, 'wb'))

    # policy
    filename = f'simulation_{simulation_id}_{args.model}_policy.pkl'
    filepath = os.path.join(output_dir, filename)
    pickle.dump(policy, open(filepath, 'wb'))   

    # policy_inputs
    filename = f'simulation_{simulation_id}_{args.model}_policy_inputs.pkl'
    filepath = os.path.join(output_dir, filename)
    pickle.dump(policy_inputs, open(filepath, 'wb'))

    # sim
    filename = f'simulation_{simulation_id}_{args.model}_sim.pkl'
    filepath = os.path.join(output_dir, filename)
    pickle.dump(sim, open(filepath, 'wb'))
