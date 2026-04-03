# IMDP Abstraction with Online MPC Control Optimization

## Overview

This repository contains the experimental framework used to produce the results developed in the paper: 
- "A. Riccardi, T. Badings, L. Laurenti, A. Abate, and B. De Schutter, *Temporal Logic Control of Nonlinear Stochastic Systems with Online Performance Optimization*, [[LINK]](https://arxiv.org/abs/2604.01372)". 

This software repository can be referenced using: 
- "A. Riccardi, T. Badings, L. Laurenti, A. Abate, and B. De Schutter, *Code for publication: Temporal Logic Control of Nonlinear Stochastic Systems with Online Performance Optimization*, DOI: 10.4121/631b574d-40e8-4951-b3ac-e304e3f34b13"

The repository is developed extending the framework previously developed  in **IMDP abstraction procedure** from [ReadMeAbstraction.md](ReadMeAbstraction.md) by integrating **online control optimization** using **Mixed Integer Programming (MIP)** and **hybrid systems theory**. The new framework combines abstraction-based robust policy synthesis for the satisfaction of control specifications with online Model Predictive Control (MPC) to introduce the first model-based performance optimization of abstraction-based policies control for nonlinear stochastic systems.

The overall architecture is constituted of **two main steps**, the first is offline, and the second is online:

1. **Formal abstraction and policy synthesis** (offline): Extends the original IMPD abstraction procedure by introducing parameters **epsilon** which allows the definition of Lp-balls, i.e., of optimization spaces surrouding the nominal actions defined by the robust policies and for which the satisfaction of the control specification is guaranteed according to a probability threshold. Such Lp-balls constitute the optimization spaces from which the control action is then selected online. 
2. **Online MPC Control Loop** (online): Implements a feedback controller using MPC that selects control actions from the Lp-balls associated with the actions defined in the robust policy synthesis. The selection is guided by predictions models constructed using Piecewise Affine (PWA) approximations of nonlinear dynamics. The optimization-based approach is therefore logic-driven and based on hybrid systems theory, and uses Mixed Integer Programming. 

Mathematical detail of the framework, as well as the overall algorithm integrating the offline and online procedure is further explained in the paper. In this [ReadMe.md](ReadMe.md) file, practical information on how to install and use the repository is presented, together with detail about the reproduction of the results used in the paper.   

---

## Architecture

### Stage 1: Formal Abstraction and Robust Policy Synthesis (Offline)

The abstraction phase generates an IMDP model of a nonlinear stochastic system, extended with:

- **Input optimization epsilon (`epsilons`)**: Allows selecting the size of the Lp-balls that will be used for online control. The larger these spaces are, the lowest is the satisfaction probability of the control specification. Trial and error experiments are necessary to obtain the best trade-off. Usually, such relation is nonlinear, and there is an elbow point. In these experiments, best performance are usually obtained when the satisfaction probability decrease is of about 1%

The formal verification of the system consists of the steps:
- **State space partitioning**: Discretizes the continuous state space into cells
- **Input space sampling**: Selects actions from the input space, that in this case are extended with surrounding areas for which the satisfaction of the control specification will still be guaranteed
- **Computation of transition probabilities**: Computes the forward transition probabilities for each cell-action pairs
- **Robust policy synthesis**: Using the epsilon-IMDP abstraction thus obtained, a robust policy satisfying the given control specification is computed. This policy is in practice a table associating to each cell in ste state space partitioning to an action in the input space. Introducing the epsilon parameters, now the action is not only a point in the input space but a region, from which any point can be selected while still satisfying the control specification 

This stage is performed separately for different epsilon values, but at least two epsilon-IMDP models are required:
- **Nominal system** (epsilon = 0): Generates nominal abstraction bounds, this is used for comparing online optimization with a nominal robust policy
- **Epsilon extended system** (with a desired value of epsilon): Generates abstraction with extended action spaces through epsilon used for online performance optimization

For the mathematical detail about general IMDP abstraction and robust policy synthesis we refer the user to the papers:
- T. Badings et al., “Robust control for dynamical systems with non-Gaussian noise via formal abstractions,” Journal of Artificial Intelligence Research, vol. 76, pp. 341–391, 2023.
- A. Lavaei, S. Soudjani, A. Abate, and M. Zamani, “Automated verification and synthesis of stochastic hybrid systems: A survey,” Automatica, vol. 146, pp. 1–40, 2022.




### Stage 2: MPC Control (Online)

The MPC controller uses:

- **Mixed Integer Programming (MIP)** via Gurobi solver
- **Piecewise Affine (PWA) approximations** of nonlinear dynamics
- **Abstraction-based state constraints**: Links continuous MPC variables to discrete abstraction cells
- **Feedback control loop**: Runs in real-time with prediction horizon and receding horizon control

The MPC formulation enforces:
- State transitions via PWA dynamics
- Input constraints derived from abstraction
- State abstraction consistency constraints
- Quadratic cost functions with state and input weights

---
## Requirements

- The requirements to perform the abstraction procedure are detailed in [ReadMeAbstraction.md](ReadMeAbstraction.md). Performing the abstraction requires Jax and Storm, for which implementation are available in Linux and Mac, while Windows in not supported. However, we provide data for the abstraction of three benchmark systems in the folder `abstraction_data`. The user can directly load this data and use it for online optimization, as it is done in practice in the MPC simulations. In such case, the requirements for the MPC procedure are sufficient to re-produce the results.

- For the online MPC simulation, some conventional Python libraries are required. We use as a solver Gurobi through gurobipy, which provides the best optimization performance. a user will to use a different optimizer should be write its own MIQP MPC implementation independently. Installation of gurobipy is reported in the following

### Install Gurobi Solver


```bash
pip install gurobipy
```

Then activate your Gurobi license (free academic licenses available at [Gurobi website](https://www.gurobi.com/academia/academic-license-access/))

---
## Benchmarks

We provide three benchmark systems to test the architecture. To these are associated PWA models. The user wiling to extend the repository or wishing to try different benchmarks must first consider the PWA implementation of such systems. This part is the most critical one, since it requires a good level of understanding of hybrid dynamical systems. As introductory point to the topic we suggest the readings:
- A. Bemporad and M. Morari, “Control of systems integrating logic, dynamics, and constraints,” Automatica, vol. 35, pp. 407–427, 1999.
- W. P. M. H. Heemels, B. De Schutter, and A. Bemporad, “Equivalence of hybrid dynamical models,” Automatica, vol. 37, pp. 1085–1091, 2001.



The benchmark systems that we propose are the following :

### 1. **Double Integrator** (Recommended for Testing)
- **State dimension**: 2
- **Input dimension**: 1
- **Complexity**: Low (good for debugging), linear system, no approximation needed
- **Files**:
  - Abstraction: `benchmarks/Integrators.py`
  - MPC dynamics: `mpc_core/Double_integrator_dynamics.py`

### 2. **Mountain Car**
- **Dynamics**: Nonlinear stochastic system with reach-avoid specification
- **State dimension**: 2 
- **Input dimension**: 1 
- **Complexity**: Medium
- **Files**:
  - Abstraction: `benchmarks/MountainCar.py`
  - MPC dynamics: `mpc_core/mountain_car_dynamics.py`

### 3. **Dubins Small Vehicle** 
- **Dynamics**: Nonlinear stochastic system with reach-avoid specification and obstacle
- **State dimension**: 3 
- **Input dimension**: 2
- **Complexity**: High 
- **Files**:
  - Abstraction: `benchmarks/Dubins_small.py`
  - MPC dynamics: `mpc_core/Dubins_small_dynamics.py`

---

## Quick Start

You can test the overall architecture through the following steps

### Running Abstraction Data Generation

Generate abstraction data for the Double Integrator benchmark:

```bash
python RunFileAbstraction.py --model Integrators --model_version 0 --epsilons '[0.1]'
```

This stores abstraction data in `abstraction_data/abstraction_data_double_integrator/`

### Running MPC Simulation

Run MPC control on the Double Integrator with generated abstraction data:

```bash
python RunFileMPC.py \
    --model 'Double_integrator' \
    --abstraction_data_nominal 'abstraction_data_DoubleIntegrator_01' \
    --abstraction_data 'abstraction_data_DoubleIntegrator_02' \
    --simulation_id '00'\                        
    --store_simulation_data 'True'\                          
    --plot_simulation 'True'\
    --simulation_horizon 25 \
    --prediction_horizon 3 \
    --number_experiments 5\
    '--input_weight '[[1]]'\
    '--state_weight '[[1, 0], [0, 1]]'\
    '--mean_noise '0.0'\
    '--cov_noise '[0.1]'\
    '--cov_initial_state '0.25'
```

This tuns the MPC simulation for several experiments, compares the results with a nominal policy, plots the results, and finally stores the simulation data and the plots in `mpc_simulation_data/mpc_simulation_double_integrator/`

To run different experiments or test the other benchmarks, follow the above procedure with the different parameters or specifications that are explained in the following

---

## Parameter Documentation

### RunFileMPC.py Parameters

Main simulation parameters controlling the MPC framework behavior:

#### System and Data Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model` | str | `Double_integrator` | Benchmark system to control. Options: `Double_integrator`, `Mountain_car`, `Dubins_small` |
| `--abstraction_data` | str | `abstraction_data` | File containing abstraction data with epsilon. Must be generated via `RunFileAbstraction.py` and epsilon settings|
| `--abstraction_data_nominal` | str | `abstraction_data_nominal` | File containing nominal abstraction data, with epailon = 0. Used for comparing the nominal policy with MPC optimization |
| `--simulation_id` | str | `00` | Unique identifier for storing simulation results |

#### Simulation Horizon Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--simulation_horizon` | int | `25` | Total number of time steps for the control simulation. Determines experiment duration. If the horizon is too short, then the simulation may stop before the control specification is satisfied |
| `--prediction_horizon` | int | `3` | MPC prediction horizon. Shorter horizons = faster computation but worse performance; typical range: 2-10; be aware that computational complexity is exponential in the horizon length |
| `--number_experiments` | int | `10` | Number of Monte Carlo simulations to run. Higher values for better statistical confidence |

#### Noise and Initial Condition Parameters

For these parameters, it is necessary to use the same settings used in the abstraction phase to have consistent results

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--mean_noise` | float | `0.0` | Mean of the process noise affecting system dynamics. |
| `--cov_noise` | JSON list | `[0.1]` | Covariance (diagonal elements) of process noise per dimension. Example: `[0.1]` for 1D noise, `[0.005, 0.0005]` for multi-dimensional. Must match system noise dimension |
| `--cov_initial_state` | float | `0.25` | Covariance of Gaussian initial state distribution. Controls initial position uncertainty |

#### Cost Function Weights

*Parameters for performance tuning:**

| Parameter | Type | Format | Description |
|-----------|------|--------|-------------|
| `--state_weight` | JSON matrix | `[[w11, w12], [w21, w22], ...]` | State cost weight matrix Q in cost J = Σ(x'Qx + u'Ru). 2D array with shape (n_states, n_states). Larger values penalize state deviations more. Typically diagonal with positive values. Example: `[[1, 0], [0, 1]]` for equal state penalties |
| `--input_weight` | JSON matrix | `[[w11, w12], [w21, w22], ...]` | Input cost weight matrix R in cost J = Σ(x'Qx + u'Ru). 2D array with shape (n_inputs, n_inputs). Larger values penalize control effort more. Typically diagonal. Example: `[[1]]` for single input, `[[1, 0], [0, 1]]` for two inputs |

#### PWA Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--number_pwa_regions` | int | `100` | Total number of piecewise affine regions used to approximate nonlinear dynamics. Higher values = better approximation but slower optimization. THis parameter is effective only for models using PWA approximations, unused otherwise. Typical range: 50-200 |

#### Output Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--store_simulation_data` | str | `False` | If `True`, stores simulation trajectories, states, inputs, and costs to `mpc_simulation_data/` folder for post-processing |
| `--plot_simulation` | str | `False` | If `True`, generates matplotlib plots of trajectories, state evolution, and control inputs |

### Example Configurations

#### Double Integrator (Fast, for Testing)
```bash
python RunFileMPC.py \
    --model 'Double_integrator' \
    --abstraction_data_nominal 'abstraction_data_DoubleIntegrator_01' \
    --abstraction_data 'abstraction_data_DoubleIntegrator_02' \
    --simulation_id '00' \
    --store_simulation_data 'True' \
    --plot_simulation 'True' \
    --simulation_horizon '25' \
    --prediction_horizon '3' \
    --number_experiments '3' \
    --input_weight '[[1]]' \
    --state_weight '[[1, 0], [0, 1]]' \
    --mean_noise '0.0' \
    --cov_noise '[0.1]' \
    --cov_initial_state '0.25'
```

#### Mountain Car 
```bash
python RunFileMPC.py \
    --model 'Mountain_car' \
    --abstraction_data_nominal 'abstraction_data_MountainCar_01' \
    --abstraction_data 'abstraction_data_MountainCar_02' \
    --simulation_id '01' \
    --store_simulation_data 'True' \
    --plot_simulation 'True' \
    --simulation_horizon '100' \
    --prediction_horizon '3' \
    --number_experiments '10' \
    --input_weight '[[1]]' \
    --state_weight '[[1, 0], [0, 1]]' \
    --mean_noise '0.0' \
    --cov_noise '[0.005, 0.0005]' \
    --cov_initial_state '0.05'
```

#### Dubins Small Vehicle 
```bash
python RunFileMPC.py \
    --model 'Dubins_small' \
    --abstraction_data_nominal 'abstraction_data_Dubins_small_01' \
    --abstraction_data 'abstraction_data_Dubins_small_02' \
    --simulation_id '02' \
    --store_simulation_data 'True' \
    --plot_simulation 'True' \
    --simulation_horizon '25' \
    --prediction_horizon '3' \
    --number_experiments '3' \
    --input_weight '[[1, 0], [0, 1]]' \
    --state_weight '[[1, 0, 0], [0, 1, 0], [0, 0, 1]]' \
    --number_pwa_regions '100' \
    --mean_noise '0.0' \
    --cov_noise '[0.1]' \
    --cov_initial_state '0.25'
```

---

## MPC Optimizer Parameters

The MPC uses **Gurobi Mixed Integer Solver** with the following tunable parameters (defined in `mpc_core/simulation_functions.py`):

### Optimization Model Structure

| Component | Description |
|-----------|-------------|
| **Decision Variables** | State trajectory `x̃[0:T]` (continuous), Input trajectory `ũ[0:T-1]` (continuous), Region indicator `δ[0:T]` (binary), auxiliary variables `z[0:T]` (continuous) |
| **Constraints** | PWA dynamics, logic-drive abstraction cell selection, state and input spaces constraints, cost minimization |
| **Objective** | Minimize J = Σ(x'Qx + u'Ru) subject to dynamics and constraints |

### Gurobi Solver Configuration

| Parameter | Default | Description | Range |
|-----------|---------|-------------|-------|
| **OutputFlag** | 0 | Print solver messages to console (1=on, 0=off) | 0, 1 |
| **LogToConsole** | 0 | Log Gurobi output to console | 0, 1 |
| **MIPFocus** | 3 | Optimization focus: 1=feasibility, 2=bound, 3=optimality | 1, 2, 3 |
| **MIPGap** | 0.05 | Optimality gap tolerance (5%). Smaller = more optimal but slower | 0.001-0.1 |
| **Heuristics** | 0.25 | Time spent on heuristics (0-1 scale) | 0.1-0.5 |
| **Threads** | 0 | Number of solver threads (0=auto-detect all cores) | 0, 1, 2, ... |

#### Tuning Guidance

- **For real-time requirements**: Increase `MIPGap` (e.g., 0.10) to trade optimality for speed
- **For better solutions**: Decrease `MIPGap` (e.g., 0.001) and increase `Heuristics` (e.g., 0.5)
- **For feasibility**: Set `MIPFocus = 1`
- **Parallel computation**: Set `Threads = 0` to use all available cores

Extensive guides about how to use Gurobi and how to select these parameters area available in the documentation of the solver at [Gurobi website](https://www.gurobi.com/)

---

## PWA (Piecewise Affine) Model Design

**This is a critical design choice that cannot be automated** and must be customized for each new system.

### What is a PWA Model?

A PWA approximation represents a nonlinear system as:

$$\dot{\mathbf{x}} = \mathbf{A}_i \mathbf{x} + \mathbf{B}_i \mathbf{u} + \mathbf{g}_i \quad \text{if } \mathbf{x} \in \mathcal{R}_i$$

where:
- $\mathcal{R}_i$ are disjoint regions partitioning the input-state space
- $(\mathbf{A}_i, \mathbf{B}_i, \mathbf{g}_i)$ are linear dynamics for region $i$
- Regions can be defined via convex polytopes

### How to implement a PWA Model?

In this work, we use PWA approximations of the `Mountain car` and `Small Dubins` systems. In both cases the models are nonlinear therefore we partition the state space into regions whose number is given by the parameter `--number_pwa_regions` and in these area we use a linear dynamics to approximate the nonlinear behavior. Such an approach allows predicting (forecasting) the evolution of the state of the system across the MPC optimization horizon and to select optimal control action. 

For both nonlinear systems, we have have to approximate a sine or a cosine function. For this we use linear segments that approximate the nonlinearity between two points of the PWA regions. Moreover, for the `Small Dubins` model, we also have a product between a system state and its cosine. TO approximate such a nonlinearity, we use McCormick-envelope inequalities.  

## Abstraction Data Parameters

The abstraction data generation (via `RunFileAbstraction.py`) uses these key parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `--model` | str | Benchmark system (Integrators, MountainCar, Dubins_small) |
| `--epsilons` | JSON list | Input optimization epsilon values. Controls discretization of input space. Smaller → finer discretization → better performance but larger abstraction. Example: `[0.1]` or `[0.1, 0.05]` |
| `--model_version` | int | System parameter variant (0, 1, 2 for different uncertainty levels) |
| `--store_abstraction_data` | str | If True, saves abstraction to folder |

Further detail on how to use such parameters is provided in 
[ReadMeAbstraction.md](ReadMeAbstraction.md)

---

## Workflow

### Typical Usage Workflow

#### 1. Prepare Abstraction Data
```bash
# Generate nominal abstraction 
python RunFileAbstraction.py --model Integrators --epsilons '[0.0]' \
    --simulation_id '01' --store_abstraction_data 'True'

# Generate abstraction using epsilon
python RunFileAbstraction.py --model Integrators --epsilons '[0.15]' \
    --simulation_id '02' --store_abstraction_data 'True'
```

#### 2. Configure MPC Parameters
Edit `RunFileMPC.py` or use command-line arguments to set:
- Cost weights (--state_weight, --input_weight)
- Prediction horizon (--prediction_horizon)
- Noise parameters (--cov_noise, --cov_initial_state)

#### 3. Run MPC Simulation
```bash
python python RunFileMPC.py \
    --model 'Double_integrator' \
    --abstraction_data_nominal 'abstraction_data_DoubleIntegrator_01' \
    --abstraction_data 'abstraction_data_DoubleIntegrator_02' \
    --simulation_id '00' \
    --store_simulation_data 'True' \
    --plot_simulation 'True' \
    --simulation_horizon '25' \
    --prediction_horizon '3' \
    --number_experiments '3' \
    --input_weight '[[1]]' \
    --state_weight '[[1, 0], [0, 1]]' \
    --mean_noise '0.0' \
    --cov_noise '[0.1]' \
    --cov_initial_state '0.25'
```

#### 4. Analyze Results
- Check `mpc_simulation_data/` folder for stored trajectories
- View plots in `output/` folder
- Evaluate performance metrics from terminal output

---

## Extending to New Systems

To integrate a new nonlinear system into this framework:

### Step 1: Create Benchmark Model
Add to `benchmarks/YourModel.py`:
- Nonlinear dynamics function
- State and input constraints
- Disturbance model

### Step 2: Design PWA Approximation
Create `mpc_core/YourModel_dynamics.py`:
- Partition state space into regions
- Fit affine approximations in each region
- Implement PWA matrices (A, B, g)
- Follow examples: Double_integrator, mountain_car, Dubins_small

### Step 3: Build MPC Optimization Model
Add to `mpc_core/simulation_functions.py`:
- `simulation_functions_yourmodel()` function
- Encode PWA dynamics as MIP constraints
- Define binary region indicators

### Step 4: Register in RunFileMPC.py
Update:
- Model selection logic
- Abstraction data folder mapping
- Dynamics class instantiation

### Step 5: Generate Abstraction Data
Run `RunFileAbstraction.py` with your new benchmark to generate IMDP abstraction.

### Step 6: Run MPC Simulations
Execute `RunFileMPC.py` with your system and PWA approximation.

---

## Simulations Plots

The simulation environments allows you to decide whether to store and plot the results. Standard plots at the end of each simulation are fine for most cases. However, if you want to re-make the plots using simulation data that you have obtained already, you can use the standalone scripts in the folder `paper_plots`. These scripts, divided by simulation model, allows you to fine-tune the plot parameters to get the desired editorial quality. In addition, in the subfolder `small_dubins`, there is also standalone scripts to visualize the abstraction, and to perform the analysis of the synthetic data. The scripts are simple graphical implementations, and are easy to interpret and modify for different uses. 


---

## References

This work combines:
- **IMDP Abstraction**: Formal abstraction of stochastic systems into Markov models
- **Mixed Integer Programming**: Optimization of hybrid systems via Gurobi
- **Piecewise Affine Systems**: Hybrid approximations of nonlinear dynamics
- **Mixed-Logical Dynamical Systems**: For logic-driven selection of optimization spaces
- **Model Predictive Control**: Receding horizon optimization for feedback control

See [ReadMeAbstraction.md](ReadMeAbstraction.md) for original IMDP abstraction documentation.

---


## Citation

If you use this framework in your research, please cite the following: 
- Original IMDP abstraction work (see ReadMeAbstraction.md). The reference paper for this work is:
    - T. Badings et al., “Robust control for dynamical systems with non-Gaussian noise via formal abstractions,” Journal of Artificial Intelligence Research, vol. 76, pp. 341–391, 2023.
- This extension combining abstraction with MPC:
    - For the reference paper: "A. Riccardi, T. Badings, L. Laurenti, A. Abate, and B. De Schutter, *Temporal Logic Control of Nonlinear Stochastic Systems with Online Performance Optimization*, [ADD ARXIV LINK]"
    - For this repository: "A. Riccardi, T. Badings, L. Laurenti, A. Abate, and B. De Schutter, *Code for publication: Temporal Logic Control of Nonlinear Stochastic Systems with Online Performance Optimization*, DOI: 10.4121/631b574d-40e8-4951-b3ac-e304e3f34b13"

- For references in the hybrid dynamical systems theory, PWA approximations and Mixed-Logical Dynamical systems we suggest:
    - A. Bemporad and M. Morari, “Control of systems integrating logic, dynamics, and constraints,” Automatica, vol. 35, pp. 407–427, 1999.
    - W. P. M. H. Heemels, B. De Schutter, and A. Bemporad, “Equivalence of hybrid dynamical models,” Automatica, vol. 37, pp. 1085–1091, 2001.

---

## License

GNU General Public License v3.0

---

**Last Updated**: March 2026
