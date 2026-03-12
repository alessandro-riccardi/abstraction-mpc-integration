import numpy as np

def wrap_theta(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

class DubinsSmallDynamics:

    def __init__(self,x0=None):
        
        self.tau = 1
        self.alpha = 0.85
        self.x = np.zeros(3) if x0 is None else x0

    def step(self, u):

        [pos_x, pos_y, theta] = self.x
        [u_1, u_2] = u

        pos_x_plus = pos_x + self.tau * u_2 *np.cos(theta)
        pos_y_plus = pos_y + self.tau * u_2 *np.sin(theta)
        theta_plus = wrap_theta(theta + self.tau * self.alpha * u_1)

        self.x = np.array([pos_x_plus, pos_y_plus, theta_plus])
        return self.x
    
    def get_state(self):
        return self.x
    
    def set_state(self, x_forced):
        self.x = x_forced

class DubinsSmallDynamicsStochastic:

    def __init__(self,x0=None):
        
        self.tau = 1
        self.alpha = 0.85
        self.x = np.zeros(3) if x0 is None else x0

    def step(self, u, w):

        [pos_x, pos_y, theta] = self.x
        [u_1, u_2] = u

        pos_x_plus = pos_x + self.tau * u_2 *np.cos(theta)
        pos_y_plus = pos_y + self.tau * u_2 *np.sin(theta)
        theta_plus = wrap_theta(theta + self.tau * self.alpha * u_1 + w)

        self.x = np.array([pos_x_plus, pos_y_plus, theta_plus])
        return self.x
    
    def get_state(self):
        return self.x
    
    def set_state(self, x_forced):
        self.x = x_forced

        
