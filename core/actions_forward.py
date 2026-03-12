import itertools
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@partial(jax.jit, static_argnums=(0))
def forward_reach(step_set, state_min, state_max, input, epsilons, cov_diag, number_per_dim, cell_width, boundary_lb, boundary_ub):
    """
    Computes the forward reachable set given a set of input parameters.

    :param step_set: Function that computes the minimum and maximum reachable states given the state bounds and input.
    :param state_min: Lower bound of the box (of states ) to propagate.
    :param state_max: Upper bound of the box (of states ) to propagate.
    :param input: Control input for the dynamical system.
    :param cov_diag: Diagonal entries of the covariance matrix
    :param number_per_dim: The number of cells per dimension in the state space grid.
    :param cell_width: The width of cells along each dimension.
    :param boundary_lb: The lower bound of the grid of the state space.
    :param boundary_ub: The upper bound of the grid of the state space.
    :return: A tuple containing:
        - frs_min: The minimum bound of the forward reachable set.
        - frs_max: The maximum bound of the forward reachable set.
        - frs_span: The number of grid cells encompassed by the forward reachable set.
        - idx_low: The lower index bounds in the grid corresponding to the forward reachable set.
        - idx_upp: The upper index bounds in the grid corresponding to the forward reachable set.
    """
    # frs_min, frs_max = step_set(state_min, state_max, input - epsilon, input + epsilon)
    
    # input_min = input-0.1
    # input_max = input+0.1
    # frs_min, frs_max = step_set(state_min, state_max, input_min, input_max)

    frs_min, frs_max = step_set(state_min, state_max, input - epsilons, input + epsilons)
    # frs_min, frs_max = step_set(state_min, state_max, input - 0.1, input + 0.1)
    # If covariance is zero, then the span equals the number of cells the forward reachable set contains at most
    frs_span = jnp.astype(jnp.ceil((frs_max - frs_min) / cell_width), int)

    state_min_norm = (frs_min - boundary_lb) / (boundary_ub - boundary_lb) * number_per_dim
    lb_contained_in = state_min_norm // 1

    idx_low = (jnp.clip(lb_contained_in, 0, (number_per_dim - 1)) * (cov_diag == 0)).astype(int)
    idx_upp = (jnp.clip(lb_contained_in + frs_span - 1, 0, number_per_dim - 1) * (cov_diag == 0) + (number_per_dim - 1) * (cov_diag != 0)).astype(int)

    return frs_min, frs_max, frs_span, idx_low, idx_upp


class RectangularForward(object):

    def __init__(self, partition, model, debug=True):
        print('Define target points and forward reachable sets...')
        t_total = time.time()

        # Vectorized function over different sets of points
        vmap_forward_reach = jax.vmap(forward_reach, in_axes=(None, None, None, 0, None, None, None, None, None, None), out_axes=(0, 0, 0, 0, 0,))

        discrete_per_dimension = [np.linspace(model.uMin[i], model.uMax[i], num=model.num_actions[i]) for i in range(len(model.num_actions))]
        discrete_inputs = np.array(list(itertools.product(*discrete_per_dimension)))
        
        t = time.time()

        # If debug is enabled, show the forward reachable set for the initial state
        if debug:
            region = partition.x2state(model.x0)[0]
            lb = partition.regions['lower_bounds'][region]
            ub = partition.regions['upper_bounds'][region]

            flb, fub, fsp, fil, fiu = vmap_forward_reach(model.step_set, lb, ub, discrete_inputs, model.epsilons, model.noise['cov_diag'], partition.number_per_dim, partition.cell_width,
                                                         partition.boundary_lb, partition.boundary_ub)
            
            print('Lower bound reachable set:\n', flb)
            print('Upper bound reachable set:\n', fub)

            self.plot_forward_reachable_sets(partition, model, model.x0, flb, fub, title='Forward Reachable Sets from Initial State')

        frs = {}
        pbar = tqdm(enumerate(zip(partition.regions['lower_bounds'], partition.regions['upper_bounds'])), total=len(partition.regions['lower_bounds']))
        self.max_slice = np.zeros(model.n)
        for i, (lb, ub) in pbar:
            # For every state, compute for every action the [lb,ub] forward reachable set
            
            flb, fub, fsp, fil, fiu = vmap_forward_reach(model.step_set, lb, ub, discrete_inputs, model.epsilons, model.noise['cov_diag'], partition.number_per_dim, partition.cell_width,
                                                         partition.boundary_lb, partition.boundary_ub)

            frs[i] = {}
            frs[i]['lb'] = flb
            frs[i]['ub'] = fub
            frs[i]['idx_lb'] = fil
            frs[i]['idx_ub'] = fiu

            self.max_slice = np.maximum(self.max_slice, jnp.max(fiu + 1 - fil, axis=0))
        self.max_slice = tuple(np.astype(self.max_slice, int).tolist())

        print(f'- Forward reachable sets computed (took {(time.time() - t):.3f} sec.)')

        self.inputs = discrete_inputs
        self.idxs = np.arange(len(discrete_inputs))
        self.frs = frs

        print(f'Defining actions took {(time.time() - t_total):.3f} sec.')
        print('')
        return

    def plot_forward_reachable_sets(self, partition, model, state, flb, fub, labels=None, title="Forward Reachable Sets"):
        """
        Plots a fixed point together with n boxes representing forward reachable sets.
        
        :param state: A point to plot (array-like of dimension 2 or 3).
        :param flb: Lower bounds of the forward reachable sets (m x n array).
        :param fub: Upper bounds of the forward reachable sets (m x n array).
        :param labels: Optional list of labels for each box.
        :param title: Title for the plot.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        state = np.array(state)
        print(state)
        dim = len(state)
        
        if dim == 2:
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Plot fixed point
            ax.plot(state[0], state[1], 'ro', markersize=10, label='Fixed Point', zorder=10)
            
            # Plot boxes
            for idx in range(flb.shape[0]):
                flb_i = flb[idx]
                fub_i = fub[idx]
                width = fub_i[0] - flb_i[0]
                height = fub_i[1] - flb_i[1]
                
                label = labels[idx] if labels is not None and idx < len(labels) else f'Box {idx}'
                rect = Rectangle((flb_i[0], flb_i[1]), width, height, 
                                linewidth=2, edgecolor=f'C{idx}', facecolor=f'C{idx}', 
                                alpha=0.3, label=label)
                ax.add_patch(rect)
            
            ax.set_xlabel('State Dimension 1')
            ax.set_ylabel('State Dimension 2')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
        elif dim == 3:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot fixed point
            ax.scatter(state[0], state[1], state[2], 
                        c='r', marker='o', s=100, label='Fixed Point', zorder=10)
            
            # Plot boxes
            for idx in range(flb.shape[0]):
                flb_i = flb[idx]
                fub_i = fub[idx]
                
                # Create vertices of the box
                vertices = [
                    [flb_i[0], flb_i[1], flb_i[2]], [fub_i[0], flb_i[1], fub_i[2]],
                    [fub_i[0], fub_i[1], flb_i[2]], [flb_i[0], fub_i[1], flb_i[2]],
                    [flb_i[0], flb_i[1], fub_i[2]], [fub_i[0], flb_i[1], fub_i[2]],
                    [fub_i[0], fub_i[1], fub_i[2]], [flb_i[0], fub_i[1], fub_i[2]]
                ]
                
                # Define the 6 faces of the box
                faces = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]],
                    [vertices[4], vertices[5], vertices[6], vertices[7]],
                    [vertices[0], vertices[1], vertices[5], vertices[4]],
                    [vertices[2], vertices[3], vertices[7], vertices[6]],
                    [vertices[0], vertices[3], vertices[7], vertices[4]],
                    [vertices[1], vertices[2], vertices[6], vertices[5]]
                ]
                
                label = labels[idx] if labels is not None and idx < len(labels) else f'Box {idx}'
                poly = Poly3DCollection(faces, alpha=0.3, facecolor=f'C{idx}', 
                                        edgecolor=f'C{idx}', linewidth=2, label=label)
                ax.add_collection3d(poly)
            
            ax.set_xlabel('State Dimension 1')
            ax.set_ylabel('State Dimension 2')
            ax.set_zlabel('State Dimension 3')
            ax.set_title(title)
            ax.legend()
            
        else:
            raise ValueError(f"Plotting is only supported for 2D and 3D. Got dimension {dim}.")
        
        i1, i2 = np.array(model.plot_dimensions, dtype=int)
        ax.set_xlim(np.array(partition.boundary_lb)[i1], np.array(partition.boundary_ub)[i1])
        ax.set_ylim(np.array(partition.boundary_lb)[i2], np.array(partition.boundary_ub)[i2])

        plt.tight_layout()
        plt.show()
        
        return fig, ax