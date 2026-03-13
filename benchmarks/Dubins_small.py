from functools import partial

from benchmarks.models import DubinsSmallDynamics
import jax
import jax.numpy as jnp
import numpy as np

from core import setmath


def wrap_theta(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


class Dubins_small(DubinsSmallDynamics):
    '''
    Simplified version of the Dubin's vehicle benchmark, with a 3D state space and a 2D control input space.
    '''

    def __init__(self, args):
        DubinsSmallDynamics.__init__(self, args)
        
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

        self.partition = {}
        self.targets = {}

        # Authority limit for the control u, both positive and negative
        self.uMin = [-0.50 * np.pi, -3]
        self.uMax = [0.50 * np.pi, 3]
        self.num_actions = [7, 5]

        # Simulation set 1 ###########################
        # Input L_p balls range
        # SIM_ID = 00
        # self.epsilons = np.array([0.0,0.0])
        # # SIM_ID = 01
        # self.epsilons = np.array([0.01,0.02])
        # SIM_ID = 02
        # self.epsilons = np.array([0.02,0.04])
        # # SIM_ID = 03
        # self.epsilons = np.array([0.04,0.08])
        # # SIM_ID = 04
        # self.epsilons = np.array([0.05,0.01])
        # SIM_ID = 05
        # self.epsilons = np.array([0.02,0.02])
        # SIM_ID = 06
        # self.epsilons = np.array([0.01,0.01])
        ################################################

        # Simulation set 2 ###########################
        # Input L_p balls range
        # SIM_ID = 02_1
        # self.epsilons = np.array([0.0,0.0])
        # # SIM_ID = 02_2
        # self.epsilons = np.array([0.01,0.02])
        # SIM_ID = 02_3
        # self.epsilons = np.array([0.02,0.04])
        # # SIM_ID = 02_4
        # self.epsilons = np.array([0.04,0.08])
        # # SIM_ID = 02_5
        # self.epsilons = np.array([0.08,0.16])
        # # SIM_ID = 02_6
        # self.epsilons = np.array([0.07,0.14])
        # SIM_ID = 02_7
        # self.epsilons = np.array([0.16,0.32])
        ################################################

        # Simulation set 3 ###########################
        # Input L_p balls range
        # SIM_ID = 03_1
        # self.epsilons = np.array([0.0,0.0])
        # # SIM_ID = 03_2
        # self.epsilons = np.array([0.1,0.2])
        # SIM_ID = 03_3
        # self.epsilons = np.array([0.15,0.3])
        # # SIM_ID = 03_4
        # self.epsilons = np.array([0.2,0.4])
        # # SIM_ID = 03_5
        # self.epsilons = np.array([0.16,0.32])
        # # SIM_ID = 03_6
        # self.epsilons = np.array([0.17,0.34])
        # SIM_ID = 03_7
        # self.epsilons = np.array([0.18,0.36])
        # SIM_ID = 03_8
        # self.epsilons = np.array([0.19,0.38])
        ################################################

        # Simulation set 4 ###########################
        # Input L_p balls range
        # SIM_ID = 03_1
        # self.epsilons = np.array([0.0,0.0])
        # # SIM_ID = 03_2
        # self.epsilons = np.array([0.1,0.2])
        # SIM_ID = 03_3
        # self.epsilons = np.array([0.15,0.3])
        # # SIM_ID = 03_4
        # self.epsilons = np.array([0.2,0.4])
        # # SIM_ID = 03_5
        # self.epsilons = np.array([0.16,0.32])
        # # SIM_ID = 03_6
        # self.epsilons = np.array([0.17,0.34])
        # SIM_ID = 03_7
        # self.epsilons = np.array([0.18,0.36])
        # SIM_ID = 03_8
        # self.epsilons = np.array([0.19,0.38])
        ################################################

        self.partition['boundary'] = np.array([[-10, -10, -np.pi], [10, 10, np.pi]])
        self.partition['boundary_jnp'] = jnp.array(self.partition['boundary'])
        self.partition['number_per_dim'] = np.array([20, 20, 11])

        # # Simulation set 1 ############################
        # self.goal = np.array([
        #     [[5, 5, -np.pi], [10, 10, np.pi]]
        # ], dtype=float)
        # ################################################

        # Simulation set 2 ############################
        self.goal = np.array([
            [[-10, 5, -np.pi], [-5, 10, np.pi]]
        ], dtype=float)
        ################################################

        # self.critical = np.array([
        #     [[-10, -10, -np.pi], [-9, -9, np.pi]],
        # ], dtype=float)

        # self.critical = np.array([
        #     # [[-10, -6, -np.pi], [-3, -5, np.pi]],
        #     # [[-3, -6, -np.pi], [-2, -5, np.pi]],
        #     [[-1, -1, -np.pi], [6, 0, np.pi]],
        #     [[0, -1, -np.pi], [0, 4, np.pi]],
        #     [[-6, 3, -np.pi], [-1, 4, np.pi]],
        #     [[-4, 7, -np.pi], [-3, 10, np.pi]],
        #     [[4, 8, -np.pi], [5, 10, np.pi]],
        #     [[8, 4, -np.pi], [10, 5, np.pi]],
        #     [[5, -4, -np.pi], [6, -1, np.pi]]
        # ], dtype=float)

        # self.critical = np.array([
        #     [[-10, -5, -np.pi], [2, -4, np.pi]],
        #     [[1, -6, -np.pi], [2, -5, np.pi]],
        #     [[-1, -1, -np.pi], [6, 0, np.pi]],
        #     [[-1, 0, -np.pi], [0, 4, np.pi]],
        #     [[-6, 3, -np.pi], [-1, 4, np.pi]],
        #     # [[-4, 7, -np.pi], [-3, 10, np.pi]],
        #     [[4, 8, -np.pi], [5, 10, np.pi]],
        #     [[8, 4, -np.pi], [10, 5, np.pi]],
        #     # [[5, -10, -np.pi], [6, -6, np.pi]]
        # ], dtype=float)

        # self.critical = np.array([
        #     [[-10, -5, -np.pi], [2, -4, np.pi]],
        #     [[-2, 2, -np.pi], [10, 3, np.pi]],
        #     [[-2, 3, -np.pi], [-1, 6, np.pi]]
        # ], dtype=float)

        # self.critical = np.array([
        #     [[-10, -5, -np.pi], [4, -4, np.pi]],
        #     [[-4, 2, -np.pi], [10, 3, np.pi]],
        #     [[-4, 3, -np.pi], [-3, 6, np.pi]]
        # ], dtype=float)

        # Simulation set 1 #############################
        # self.critical = np.array([
        #     [[-10, -5, -np.pi], [2, -4, np.pi]],
        #     [[-2, 2, -np.pi], [10, 3, np.pi]]
        # ], dtype=float)
        ################################################

        # # Simulation set 2 #############################
        # self.critical = np.array([
        #     [[-10, -1, -np.pi], [-1, 1, np.pi]],
        #     [[-1, -5, -np.pi], [1, 5, np.pi]]
        # ], dtype=float)
        # ################################################

        # Simulation set 3 #############################
        self.critical = np.array([
            [[-10, -1, -np.pi], [-1, 1, np.pi]],
            [[-1, -3, -np.pi], [1, 1, np.pi]]
        ], dtype=float)
        ################################################

        # self.critical = np.array([
        #     [[-10, -5, -np.pi], [2, -4, np.pi]],
        #     [[-2, 2, -np.pi], [10, 3, np.pi]],
        #     # [[1, -6, -np.pi], [2, -3, np.pi]],
        #     # [[-2, 1, -np.pi], [-1, 5, np.pi]],
        #     # [[7, -5, -np.pi], [10, -4, np.pi]],
        #     # [[-10, 4, -np.pi], [-7, 5, np.pi]]
        # ], dtype=float)


        # self.x0 = np.array([-5, 5, 0])

        # Simulation set 1 #############################
        # self.x0 = np.array([-9.5, -8.5, 0])
        ################################################

        # Simulation set 2 #############################
        # self.x0 = np.array([-7.5, -7.5, 0])
        ################################################

        # Simulation set 3 #############################
        self.x0 = np.array([-9.5, -2.5, 0])
        ################################################

        # self.x0 = np.array([-7.5, -7.5, 0])

        return
