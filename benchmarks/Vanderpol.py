from functools import partial
from benchmarks.models import VanderpolDynamics
import jax
import jax.numpy as jnp
import numpy as np
from core import setmath
from matplotlib import animation
import numpy as onp
from matplotlib.patches import Polygon, Circle
import math
from matplotlib import pyplot as plt


class Vanderpol(VanderpolDynamics):
    '''
    Pendulum benchmark.
    '''

    def __init__(self, args):
        VanderpolDynamics.__init__(self, args)

        self.plot_dimensions = [0, 1]

        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.lump = 1

        self.set_spec()
        print('')

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.

        Coincides with Problem 3 (Quantitative Reachability for the Van der Pol Oscillator of ARCH_COMP22)
        '''

        self.partition = {}
        self.targets = {}
        
        # Authority limit for the control u, both positive and negative
        self.uMin = [-2]
        self.uMax = [2]
        self.num_actions = [11]

        self.epsilons = 0 * np.array([0.2])

        self.partition['boundary'] = np.array([[-4, -4], [4, 4]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([200, 100])

        self.goal = np.array([
            # [[-1.2, -2.9], [-0.9, -2]]
            [[0.5,1], [1.5,4]]
        ], dtype=float)

        self.critical = np.array([
        ], dtype=float)

        self.x0 = np.array([1, -2])

        return