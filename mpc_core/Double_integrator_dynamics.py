import numpy as np

class DoubleIntegratorDynamics:

    def __init__(self,x0=None):
        
        self.STATES_NUMBER = 2
        self.INPUTS_NUMBER = 2
        self.NOISE_NUMBER = 2

        # Discretization step size
        self.tau = 1.0

        # State transition matrix
        self.A  = np.array([[1, self.tau],
                            [0, 1]])
        
        # Input matrix
        self.B  = np.array([[self.tau**2/2],
                            [self.tau]])
    
        # Disturbance matrix
        self.Q  = np.array([[0],[0],])
        self.x = np.zeros(2) if x0 is None else x0

    def step(self, action, noise):
        state_next = self.A @ self.x + self.B @ action + noise
        self.x = state_next
        return self.x
    
    def get_state(self):
        return self.x
    
    def set_state(self, x_forced):
        self.x = x_forced


        
