############################################

## Double Integrator

# SIM_ID = 01
# self.epsilons = np.array([0.0])
- Verify with storm took: 0.062 sec.
Total sum of reach probs: 376.73178353999225
In state [ 0 -8]: [0.99869584]
Empirical satisfaction probability: 1.0

# SIM_ID = 02
self.epsilons = np.array([0.1])
- Verify with storm took: 0.524 sec.
Total sum of reach probs: 372.6610773720503
In state [ 0 -8]: [0.99216345]

# SIM_ID = 03
# self.epsilons = np.array([0.2])
- Verify with storm took: 2.632 sec.
Total sum of reach probs: 338.83742961790927
In state [ 0 -8]: [0.89605999]

# SIM_ID = 04
# self.epsilons = np.array([0.3])
- Verify with storm took: 2.765 sec.
Total sum of reach probs: 335.597979939126
In state [ 0 -8]: [0.89179643]

# SIM_ID = 05
# self.epsilons = np.array([0.4])
- Verify with storm took: 3.406 sec.
Total sum of reach probs: 331.20604681413704
In state [ 0 -8]: [0.88328404]

# SIM_ID = 06
self.epsilons = np.array([0.5])
- Verify with storm took: 5.450 sec.
Total sum of reach probs: 335.8039974842354
In state [ 0 -8]: [0.89789405]

# SIM_ID = 07
self.epsilons = np.array([1])
- Verify with storm took: 36.581 sec.
Total sum of reach probs: 268.5824667373293
In state [ 0 -8]: [0.7288505]

# SIM_ID = 08
# self.epsilons = np.array([2])
- Verify with storm took: 123.231 sec.
Total sum of reach probs: 131.72198898049604
In state [ 0 -8]: [0.38762873]


############################################

## MOUNTAIN CAR

# Nominal noise
cov = [0.005,0.0005]

# SIM_ID = 01
self.epsilons = np.array([0])
- Verify with storm took: 4.045 sec.
Total sum of reach probs: 10411.307186931705
In state [-0.6  0. ]: [0.99106904]

# SIM_ID = 02
# self.epsilons = np.array([0.1])
- Verify with storm took: 10.478 sec.
Total sum of reach probs: 10073.711635093598
In state [-0.6  0. ]: [0.98649695]

# SIM_ID = 03
# self.epsilons = np.array([0.2])
- Verify with storm took: 164.427 sec.
Total sum of reach probs: 8272.823134083632
In state [-0.6  0. ]: [0.74398564]

# SIM_ID = 04
# self.epsilons = np.array([0.15])
- Verify with storm took: 34.015 sec.
Total sum of reach probs: 9770.84906512154
In state [-0.6  0. ]: [0.96399028]

# Reduced noise
cov = [0.001,0.0001]

# SIM_ID = 05
self.epsilons = np.array([0])
- Verify with storm took: 0.681 sec.
Total sum of reach probs: 10630.62033383952
In state [-0.6  0. ]: [0.99399028]

# SIM_ID = 06
# self.epsilons = np.array([0.1])
- Verify with storm took: 1.802 sec.
Total sum of reach probs: 10211.452800826835
In state [-0.6  0. ]: [0.99044271]

# SIM_ID = 07
# self.epsilons = np.array([0.2])
- Verify with storm took: 2.822 sec.
Total sum of reach probs: 9932.808148030459
In state [-0.6  0. ]: [0.97923043]


# SIM_ID = 08
# self.epsilons = np.array([0.15])
- Verify with storm took: 1.280 sec.
Total sum of reach probs: 10109.932988578068
In state [-0.6  0. ]: [0.98949685]


############################################

## SMALL DUBINS

- Generating abstraction took: 16.499 sec.

# Noiminal noise set
cov = 0.1

SIMULATION_ID = "01"
# self.epsilons = np.array([0.0,0.0])
- Verify with storm took: 2.257 sec.
Total sum of reach probs: 3977.2907779668485
In state [-9.5 -2.5  0. ]: [0.9963738]
Empirical satisfaction probability: 1.0

SIMULATION_ID = "02"
self.epsilons = np.array([0.1,0.2])
- Verify with storm took: 5.012 sec.
Total sum of reach probs: 3872.4469523427706
In state [-9.5 -2.5  0. ]: [0.99455271]


SIMULATION_ID = "03"
self.epsilons = np.array([0.15,0.3])
- Verify with storm took: 14.870 sec.
Total sum of reach probs: 3847.2192129703067
In state [-9.5 -2.5  0. ]: [0.99134497]
Starting Monte Carlo simulations...

SIMULATION_ID = "04"
# self.epsilons = np.array([0.2,0.4])
- Verify with storm took: 2506.027 sec.
Total sum of reach probs: 2588.4239628805717
In state [-9.5 -2.5  0. ]: [0.11975548]
Starting Monte Carlo simulations...

SIMULATION_ID = "05"
self.epsilons = np.array([0.16,0.32])
- Verify with storm took: 312.616 sec.
Total sum of reach probs: 3575.1779667056526
In state [-9.5 -2.5  0. ]: [0.91418181]

SIMULATION_ID = "06"
# self.epsilons = np.array([0.17,0.34])
- Verify with storm took: 387.116 sec.
Total sum of reach probs: 3514.814668168989
In state [-9.5 -2.5  0. ]: [0.89590211]
Starting Monte Carlo simulations...

SIMULATION_ID = "07"
# self.epsilons = np.array([0.18,0.36])
- Verify with storm took: 762.225 sec.
Total sum of reach probs: 3245.3494235441985
In state [-9.5 -2.5  0. ]: [0.74051056]
Starting Monte Carlo simulations...


# REDUCED NOISE SET
cov = 0.01

SIMULATION_ID = "08"
# self.epsilons = np.array([0.10,0.20])
- Verify with storm took: 0.916 sec.
Total sum of reach probs: 3894.1299481693204
In state [-9.5 -2.5  0. ]: [0.99641743]

SIMULATION_ID = "09"
# self.epsilons = np.array([0.15,0.30])
- Verify with storm took: 1.690 sec.
Total sum of reach probs: 3893.589009724328
In state [-9.5 -2.5  0. ]: [0.9960548]

SIMULATION_ID = "10"
# self.epsilons = np.array([0.2,0.40])
- Verify with storm took: 169.017 sec.
Total sum of reach probs: 3713.734772710529
In state [-9.5 -2.5  0. ]: [0.94644075]

SIMULATION_ID = "15"
# self.epsilons = np.array([0.0,0.0])
- Verify with storm took: 0.862 sec.
Total sum of reach probs: 3978.50006829317
In state [-9.5 -2.5  0. ]: [0.99661436]

# REDUCED NOISE SET
cov = 0.05

SIMULATION_ID = "11"
# self.epsilons = np.array([0.10,0.20])
- Verify with storm took: 2.421 sec.
Total sum of reach probs: 3891.7443448351987
In state [-9.5 -2.5  0. ]: [0.99617111]
Starting Monte Carlo simulations...


SIMULATION_ID = "12"
# self.epsilons = np.array([0.15,0.30])
- Verify with storm took: 12.046 sec.
Total sum of reach probs: 3880.171855050901
In state [-9.5 -2.5  0. ]: [0.9952907]

SIMULATION_ID = "13"
# self.epsilons = np.array([0.2,0.40])
- Verify with storm took: 379.692 sec.
Total sum of reach probs: 3530.539141377445
In state [-9.5 -2.5  0. ]: [0.90330905]

SIMULATION_ID = "14"
# self.epsilons = np.array([0.0,0.0])
- Verify with storm took: 0.925 sec.
Total sum of reach probs: 3978.4984380900187
In state [-9.5 -2.5  0. ]: [0.99661469]

############################################

