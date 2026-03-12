import numpy as np
import cvxpy as cp


class MountainCarDynamics:

    def __init__(self,x0=None):

        
        self.linear = False

        self.n = 2
        self.p = 1
        self.state_variables = ['position', 'velocity']

        self.x = np.zeros(self.n) if x0 is None else x0

        # Discretization step size
        self.tau = 2

        # Parameters
        self.max_speed = 0.07
        self.gravity = 0.0025
        self.power = 0.0015

        # Covariance of the process noise
        cov = [0.005,0.0005] #[0.01, 0.001]
        self.noise = {
            'cov': np.diag(cov),
            'cov_diag': np.array(cov)
        }

    def step(self, action, noise):

        position, velocity = self.x

        velocity_plus = velocity + self.tau * (action[0] * self.power - self.gravity * np.cos(3 * position))
        velocity_plus = np.clip(velocity_plus, -self.max_speed+1e-4, self.max_speed-1e-4)
        position_plus = position + self.tau * velocity_plus + noise[0]
        velocity_plus += noise[1]

        self.x = np.array([position_plus, velocity_plus])
        return self.x
    
    def get_state(self):
        return self.x
    
    def set_state(self, x_forced):
        self.x = x_forced