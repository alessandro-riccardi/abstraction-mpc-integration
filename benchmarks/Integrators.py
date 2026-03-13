from functools import partial
from benchmarks.models import DoubleIntegratorDynamics, TripleIntegratorDynamics
import jax
import jax.numpy as jnp
import numpy as np
import scipy 
from core import setmath


class DoubleIntegrator(DoubleIntegratorDynamics):
    
    def __init__(self, args):
        DoubleIntegratorDynamics.__init__(self, args)

        self.plot_dimensions = [0, 1]

        self.epsilons = np.array(args.epsilons)

        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.lump = 1

        self.set_spec()
        print('')

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''

        # Simulation set 1 ###########################
        # Input L_p balls range
        # SIM_ID = 01
        # self.epsilons = np.array([0])
        # SIM_ID = 02
        # self.epsilons = np.array([0.1])
        # SIM_ID = 03
        # self.epsilons = np.array([0.2])
        # SIM_ID = 04
        # self.epsilons = np.array([0.3])
        # SIM_ID = 05
        # self.epsilons = np.array([0.4])
        # SIM_ID = 06
        # self.epsilons = np.array([0.5])
        # SIM_ID = 07
        # self.epsilons = np.array([1])
        # SIM_ID = 08
        # self.epsilons = np.array([2])
        ################################################

        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-5]
        self.uMax = [5]
        self.num_actions = [21]

        self.partition['boundary'] = np.array([[-21, -10.5], [21, 10.5]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([21, 21])

        self.goal = np.array([
            [[-4, -2], [4, 2]]
        ], dtype=float)

        self.critical = np.array([
        ], dtype=float)

        self.x0 = np.array([0, -8])

        return


class TripleIntegrator(TripleIntegratorDynamics):
    
    def __init__(self, args):
        TripleIntegratorDynamics.__init__(self, args)

        self.plot_dimensions = [0, 1]

        # Set value of delta (how many time steps are grouped together)
        # Used to make the model fully actuated
        self.lump = 1

        self.set_spec()
        print('')

    def set_spec(self):
        '''
        Set the abstraction parameters and the reach-avoid specification.
        '''

        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-5]
        self.uMax = [5]
        self.num_actions = [11]

        self.epsilons = np.array([0])

        self.partition['boundary'] = np.array([[-21, -21, -21], [21, 21, 21]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([21, 21, 21])

        self.goal = np.array([
            [[-8, -8, -8], [8, 8, 8]]
        ], dtype=float)

        self.critical = np.array([
        ], dtype=float)

        self.x0 = np.array([-14, 0, 0])

        return
