import numpy as np
import matplotlib.pyplot as plt

iterations = 1  # number of experiments
horizon = 25 # length of the simulation

cov = np.array([0.005,0.0005]) # the value I passed in the abstraction code fo mountain car
model_size = 2 # number of states


# Noise sequence 1
np.random.seed(0)
# not working cause: ValueError: cov must be 2 dimensional and square
# noise_one = np.random.multivariate_normal(np.zeros(model_size), 
#                                           cov ** 2, 
#                                           (iterations, horizon))
noise_one = np.random.multivariate_normal(np.zeros(model_size), 
                                          np.diag(cov), 
                                          (iterations, horizon))



noise_two = [] 
np.random.seed(0)
for j in range(0, model_size):
    # Original fomulation
    # std_dev_noise = cov[j]**2

    # Corrected formulation
    # std_dev_noise = cov[j]  

    # New formulation
    std_dev_noise = np.sqrt(cov[j])
    
    noise_two.append(np.random.normal(loc=0, scale=std_dev_noise, size=horizon))


# %%
steps = np.arange(0,horizon)

plt.plot(steps, noise_one[0,:,0])
plt.plot(steps, noise_two[0])

# plt.plot(steps, noise_one[0,:,1])
# plt.plot(steps, noise_two[1])
# %%
