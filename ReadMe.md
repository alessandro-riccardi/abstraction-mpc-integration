# IMDP Abstraction with Online MPC Control Optimization

## Overview

This repository extends the **IMDP (Interval Markov Decision Process) abstraction procedure** from [ReadMeAbstraction.md](ReadMeAbstraction.md) by integrating **online control optimization** using **Mixed Integer Programming (MIP)** and **hybrid systems theory**. The framework combines formal abstraction-based reasoning with practical model predictive control (MPC) to achieve performance-optimized control for nonlinear systems.

The key innovation is a **two-stage procedure**:

1. **Abstraction Data Generation**: Extends the original repository's abstraction procedure with parameters **epsilon** which allows the definition of Lp-balls that will be used for the definition of the online input optimiztion space
2. **Online MPC Control Loop**: Implements a feedback controller using MPC with Piecewise Affine (PWA) approximations of nonlinear dynamics, and a logic-driven seleciton of optimal policy actions (obtained through abstraction) for the optimization of a cost function

---

## Architecture

### Stage 1: Abstraction Data Generation

The abstraction phase generates an IMDP model of a nonlinear system, extended with:

- **Input optimization epsilon (`epsilons`)**: Allows selecting the size of the Lp balls that will be used for online control. The larger these spaces are, the lowest is the satisfaction probability of the control specification. Trial and error experiments are necessary to obtain the best trade-off. Usually, such relation is nonlinear, and there is an elbow point. In these experiments, best performance are usually obtained when the satisfaction probability deceease is of about 1% 
- **State space partitioning**: Discretizes the continuous state space into cells
- **Transition probabilities**: Computes robust abstractions considering system dynamics and disturbances
- **Output format**: Stores abstraction data for use in MPC

This stage is performed separately for:
- **Nominal system**: Generates nominal abstraction bounds, this is used for comparing online optimization with a nominal robust policy
- **Epsilon extended system**: Generates abstraction with extended action spaces through epsilon used for online performance optimization

-- > Arrived here in correcting the draft

### Stage 2: Online MPC Control

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

## Benchmarks

Three benchmark systems are implemented with PWA models:

### 1. **Double Integrator** (Recommended for Testing)
- **Dynamics**: `ẋ₁ = x₂`, `ẋ₂ = u`
- **State dimension**: 2
- **Input dimension**: 1
- **Complexity**: Low (good for debugging)
- **PWA model**: Naturally linear, no approximation needed
- **Files**:
  - Abstraction: `benchmarks/Integrators.py`
  - MPC dynamics: `mpc_core/Double_integrator_dynamics.py`

### 2. **Mountain Car**
- **Dynamics**: Nonlinear underactuated system with gravity and velocity saturation
- **State dimension**: 2 (position, velocity)
- **Input dimension**: 1 (throttle command)
- **Complexity**: Medium
- **PWA model**: Piecewise linear approximation of cos(3·position) nonlinearity
- **Files**:
  - Abstraction: `benchmarks/MountainCar.py`
  - MPC dynamics: `mpc_core/mountain_car_dynamics.py`

### 3. **Dubins Small Vehicle** (Benchmark Vehicle Dynamics)
- **Dynamics**: Nonlinear steering dynamics with position and orientation
- **State dimension**: 3 (x, y, θ)
- **Input dimension**: 2 (heading rate, forward velocity)
- **Complexity**: High (nonlinear angle dynamics)
- **PWA model**: Piecewise linear approximation of cos(θ) and sin(θ) nonlinearities
- **Files**:
  - Abstraction: `benchmarks/Dubins_small.py`
  - MPC dynamics: `mpc_core/Dubins_small_dynamics.py`

---

## Installation

### 1. Create Python Environment

Create a Python 3.12 environment (tested version):

```bash
conda create -n mpc-abstraction python=3.12
conda activate mpc-abstraction
```

### 2. Install System Dependencies

Install CDD library and GMP (required for pycddlib):

**macOS:**
```bash
brew install cddlib gmp
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libcdd-dev libgmp-dev
```

**Windows:**
Follow [pycddlib installation guide](https://pycddlib.readthedocs.io/en/latest/quickstart.html)

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
pip install pycddlib
```

### 4. Install JAX (Optional but Recommended)

For GPU acceleration (requires CUDA):
```bash
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
```

For CPU only:
```bash
pip install jax==0.8.0
```

### 5. Install Gurobi Solver

Gurobi is required for MPC optimization. Install via:

```bash
conda install -c gurobi gurobi
```

Then activate your Gurobi license (free academic licenses available at [Gurobi website](https://www.gurobi.com/academia/academic-license-access/))

---

## Quick Start

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
    --simulation_horizon 25 \
    --prediction_horizon 3 \
    --number_experiments 5
```

---

## Parameter Documentation

### RunFileMPC.py Parameters

Main simulation parameters controlling the MPC framework behavior:

#### System and Data Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model` | str | `Double_integrator` | Benchmark system to control. Options: `Double_integrator`, `Mountain_car`, `Dubins_small` |
| `--abstraction_data` | str | `abstraction_data` | Folder containing robust abstraction data (with uncertainty margins). Must be generated via `RunFileAbstraction.py` |
| `--abstraction_data_nominal` | str | `abstraction_data_nominal` | Folder containing nominal abstraction data (without uncertainty margins). Used for informational purposes in robust MPC. If not provided, uses `--abstraction_data` |
| `--simulation_id` | str | `00` | Unique identifier for storing simulation results and logs |

#### Simulation Horizon Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--simulation_horizon` | int | `25` | Total number of time steps for the control simulation. Determines experiment duration |
| `--prediction_horizon` | int | `3` | MPC prediction horizon (steps ahead for optimization). Shorter horizons = faster computation but less foresight; typical range: 2-10 |
| `--number_experiments` | int | `10` | Number of Monte Carlo simulations to run. Higher values for better statistical confidence |

#### Noise and Initial Condition Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--mean_noise` | float | `0.0` | Mean of the process noise affecting system dynamics. Typically 0 for mean-zero disturbances |
| `--cov_noise` | JSON list | `[0.1]` | Covariance (diagonal elements) of process noise per dimension. Example: `[0.1]` for 1D noise, `[0.005, 0.0005]` for multi-dimensional. Must match system noise dimension |
| `--cov_initial_state` | float | `0.25` | Covariance of Gaussian initial state distribution. Controls initial position uncertainty |

#### Cost Function Weights

**Critical parameters for performance tuning:**

| Parameter | Type | Format | Description |
|-----------|------|--------|-------------|
| `--state_weight` | JSON matrix | `[[w11, w12], [w21, w22], ...]` | State cost weight matrix Q in cost J = Σ(x'Qx + u'Ru). 2D array with shape (n_states, n_states). Larger values penalize state deviations more. Typically diagonal with positive values. Example: `[[1, 0], [0, 1]]` for equal state penalties |
| `--input_weight` | JSON matrix | `[[w11, w12], [w21, w22], ...]` | Input cost weight matrix R in cost J = Σ(x'Qx + u'Ru). 2D array with shape (n_inputs, n_inputs). Larger values penalize control effort more. Typically diagonal. Example: `[[1]]` for single input, `[[1, 0], [0, 1]]` for two inputs |

#### PWA Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--number_pwa_regions` | int | `100` | Total number of piecewise affine regions used to approximate nonlinear dynamics. Higher values = better approximation but slower optimization. Only used for Dubins_small. Typical range: 50-200 |

#### Output Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--store_simulation_data` | str | `False` | If `True`, stores simulation trajectories, states, inputs, and costs to `mpc_simulation_data/` folder for post-processing |
| `--plot_simulation` | str | `False` | If `True`, generates matplotlib plots of trajectories, state evolution, and control inputs |

### Example Configurations

#### Double Integrator (Fast Testing)
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

#### Mountain Car (Medium Complexity)
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

#### Dubins Small Vehicle (Complex with PWA)
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

The MPC uses **Gurobi Mixed Integer Solver** with the following tunable parameters (defined in `mpc_core/optimization_model_builders.py`):

### Optimization Model Structure

| Component | Description |
|-----------|-------------|
| **Decision Variables** | State trajectory `x̃[0:T+1]` (continuous), Input trajectory `ũ[0:T-1]` (continuous), Region indicator `δ[0:T]` (binary), Reconstruction variables `z[0:T]` (continuous) |
| **Constraints** | PWA dynamics, abstraction cell membership, input feasibility from abstraction, cost minimization |
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

---

## PWA (Piecewise Affine) Model Design

**This is a critical design choice that cannot be automated** and must be customized for each new system.

### What is a PWA Model?

A PWA approximation represents a nonlinear system as:

$$\dot{\mathbf{x}} = \mathbf{A}_i \mathbf{x} + \mathbf{B}_i \mathbf{u} + \mathbf{g}_i \quad \text{if } \mathbf{x} \in \mathcal{R}_i$$

where:
- $\mathcal{R}_i$ are disjoint regions partitioning the state space
- $(\mathbf{A}_i, \mathbf{B}_i, \mathbf{g}_i)$ are linear dynamics for region $i$
- Regions can be defined via convex polytopes

### Implemented PWA Models

#### 1. Double Integrator (No Approximation Needed)
```python
# File: mpc_core/Double_integrator_dynamics.py

# Exact linear dynamics - no PWA needed
A = [[1, τ],        B = [[τ²/2],
     [0, 1]]            [τ]]
```

This system is naturally linear, so the MPC uses exact dynamics.

#### 2. Mountain Car (Gravity Nonlinearity)
```python
# File: mpc_core/mountain_car_dynamics.py

# Nonlinear term: cos(3·position)
# PWA Approximation: Partition position into M regions
# Within each region: cos(3·p) ≈ c₀ + c₁·p (linear approximation)

# Key parameters:
- max_speed = 0.07         # Velocity saturation
- gravity = 0.0025         # Gravity coefficient
- power = 0.0015           # Actuator power
- tau = 2                  # Discretization step
```

**Design Approach**:
1. Identify nonlinear terms (cos, sin, products, etc.)
2. Partition state space into M regions (typically 50-200 for smooth functions)
3. Fit affine approximation `Ai, Bi, gi` within each region via regression or polytope fitting

#### 3. Dubins Small Vehicle (Trigonometric Nonlinearities)
```python
# File: mpc_core/Dubins_small_dynamics.py

# Nonlinear terms: cos(θ), sin(θ)
# PWA Approximation: Partition angle θ into M regions
# Within each region: cos(θ) ≈ c₀ + c₁·θ, sin(θ) ≈ s₀ + s₁·θ

# Key parameters:
- tau = 1                  # Discretization step
- alpha = 0.85             # Heading rate gain
- number_pwa_regions = 100 # Regions for θ partitioning (tunable via command line)
```

**Design Approach**:
1. Partition angle θ ∈ [−π, π] into 100 regions
2. For region i: fit linear approximation of cos, sin in that range
3. Construct Ai, Bi for each angle region
4. MPC solver chooses appropriate region based on current state

### Designing PWA Models for New Systems

To extend this framework to a new nonlinear system:

#### Step 1: Identify Nonlinearities
Extract nonlinear terms from your system dynamics. Examples:
- cos(x), sin(x): Partition state and fit affine approximation
- x₁·x₂: Switch-based regions or product approximations
- arctan(x): Bounded function, partition and approximate

#### Step 2: Choose Partitioning Strategy
Options for partitioning:
- **Uniform grids**: Equally spaced regions (simple, may be inefficient)
- **Adaptive grids**: More regions where nonlinearity is high (better approximation)
- **Polytope-based**: Use convex regions from abstraction process

#### Step 3: Fit Affine Approximations
For each region Ri:

$$[\mathbf{A}_i, \mathbf{B}_i, \mathbf{g}_i] = \arg\min_{A,B,g} \int_{R_i} \|\dot{\mathbf{x}} - (\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u} + \mathbf{g})\|^2 d\mathbf{x}$$

Methods:
- **Least squares regression**: Evaluate dynamics at sample points in region, fit affine model
- **Polytope fitting**: Use convex polytope approximation from abstraction data
- **Library methods**: Use tools like pycddlib for automatic polytope generation

#### Step 4: Implement in MPC Dynamics File
Create a new file `mpc_core/YourModel_dynamics.py`:

```python
import numpy as np

class YourModelDynamics:
    def __init__(self, x0=None):
        self.STATES_NUMBER = n_x        # State dimension
        self.INPUTS_NUMBER = n_u        # Input dimension
        self.NOISE_NUMBER = n_w         # Noise dimension
        self.tau = 1.0                  # Time discretization
        
        # Define regions for PWA
        self.pwa_regions = [...]        # Region definitions
        
        # Affine approximations for each region
        self.A = [A_1, A_2, ...]        # State transition matrices
        self.B = [B_1, B_2, ...]        # Input matrices
        self.g = [g_1, g_2, ...]        # Affine offsets
        
        self.x = np.zeros(n_x) if x0 is None else x0
    
    def step(self, action, noise):
        # Nonlinear dynamics step for simulation
        # Used in Monte Carlo validation
        x_next = self.nonlinear_dynamics(self.x, action, noise)
        self.x = x_next
        return self.x
    
    def nonlinear_dynamics(self, x, u, w):
        # Implement actual nonlinear dynamics
        # This is used for ground truth simulation
        pass
    
    def get_state(self):
        return self.x
    
    def set_state(self, x_forced):
        self.x = x_forced
```

#### Step 5: Create Optimization Model Builder
Add to `mpc_core/optimization_model_builders.py`:

```python
def optimization_model_builder_yourmodel(
    PREDICTION_HORIZON, STATES_NUMBER, INPUTS_NUMBER,
    lower_bound_x, upper_bound_x, upper_bound_u, lower_bound_u,
    NUMBER_PWA_REGIONS, centers, M_state, m_state,
    lb_z_s_k, ub_z_s_k, M_input_policy, m_input_policy
):
    """
    Build MPC model for YourModel using PWA approximation.
    
    Args:
        PREDICTION_HORIZON: Time steps for MPC
        STATES_NUMBER, INPUTS_NUMBER: Dimensions
        lower_bound_x, upper_bound_x: State constraints
        lower_bound_u, upper_bound_u: Input constraints
        NUMBER_PWA_REGIONS: Number of PWA regions
        centers: Polytope centers from abstraction
        M_state, m_state: Upper/lower bounds from abstraction
        lb_z_s_k, ub_z_s_k: Reconstruction variable bounds
        M_input_policy, m_input_policy: Input policy bounds
    
    Returns:
        MPC_model: Gurobi optimization model
    """
    # Implement PWA dynamics constraints
    # Follow structure from Double_integrator or Mountain_car examples
```

#### Step 6: Register in RunFileMPC.py
Add to the model selection logic:

```python
if model == 'YourModel':
    abstraction_folder = "abstraction_data_yourmodel"
    # Load dynamics and builder function
```

### Approximation Quality Assessment

Evaluate PWA approximation quality:

1. **Sample-based validation**: Compare nonlinear dynamics with PWA at random points
   ```python
   # Sample points in region
   # Compute nonlinear dynamics x_next_true = f(x, u)
   # Compute PWA dynamics x_next_pwa = A_i·x + B_i·u + g_i
   # Measure error: ||x_next_true - x_next_pwa||
   ```

2. **Abstraction validation**: Check if abstraction bounds are consistent
   ```python
   # Verify M_state[i], m_state[i] bounds are tight
   # If error > abstraction cell width, refine partitioning
   ```

3. **Monte Carlo simulation**: Run actual control and measure performance
   ```python
   # Execute MPC with PWA
   # Compare trajectory with nominal nonlinear simulation
   # Check constraint satisfaction and optimality gap
   ```

---

## Abstraction Data Parameters

The abstraction data generation (via `RunFileAbstraction.py`) uses these key parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `--model` | str | Benchmark system (Integrators, MountainCar, Dubins_small) |
| `--epsilons` | JSON list | Input optimization epsilon values. Controls discretization of input space. Smaller → finer discretization → better performance but larger abstraction. Example: `[0.1]` or `[0.1, 0.05]` |
| `--model_version` | int | System parameter variant (0, 1, 2 for different uncertainty levels) |
| `--store_abstraction_data` | str | If True, saves abstraction to folder |

---

## Project Structure

```
abstraction-mpc-integration/
├── ReadMe.md                     # This file
├── ReadMeAbstraction.md          # Original IMDP abstraction documentation
├── RunFileAbstraction.py         # Abstraction data generation script
├── RunFileMPC.py                 # MPC simulation and control script
├── test.py                       # Unit tests
├── requirements.txt              # Python dependencies
│
├── benchmarks/                   # Nonlinear system models for abstraction
│   ├── Integrators.py           # Double integrator benchmark
│   ├── MountainCar.py           # Mountain car benchmark
│   ├── Dubins_small.py          # Dubins vehicle benchmark
│   ├── Pendulum.py              # Other benchmarks
│   ├── Vanderpol.py
│   ├── Drone2D.py, Drone3D.py, Drone4D.py
│   └── models.py
│
├── core/                         # Abstraction generation core (original)
│   ├── imdp.py                  # IMDP model class
│   ├── model.py                 # System model abstraction
│   ├── partition.py             # State space partitioning
│   ├── polytope.py              # Polytope operations
│   ├── actions_forward.py        # Forward reachability
│   ├── simulate.py              # Simulation utilities
│   ├── utils.py                 # General utilities
│   └── options.py               # Command-line argument parsing for abstraction
│
├── mpc_core/                     # MPC framework (online control)
│   ├── Double_integrator_dynamics.py  # PWA: Double integrator
│   ├── mountain_car_dynamics.py       # PWA: Mountain car
│   ├── Dubins_small_dynamics.py       # PWA: Dubins vehicle
│   ├── optimization_model_builders.py # Gurobi MPC model construction
│   ├── mpc_support_functions.py       # Utilities (noise, reference, etc.)
│   ├── simulation_functions.py        # MPC simulation loop
│   ├── performance_evaluation.py      # Performance metrics
│   ├── plot_simulations.py           # Result visualization
│   ├── options.py                    # Command-line arguments for MPC
│   └── __pycache__/
│
├── abstraction_data/             # Generated abstraction data (output)
│   ├── abstraction_data_double_integrator/
│   ├── abstraction_data_mountain_car/
│   └── abstraction_data_small_dubin/
│
├── mpc_simulation_data/          # Simulation results (output)
│   ├── mpc_simulation_double_integrator/
│   ├── mpc_simulation_mountain_car/
│   └── mpc_simulation_small_dubin/
│
├── plotting/                     # Post-processing and visualization
│   ├── heatmap.py              # Heatmap generation
│   ├── traces.py               # Trajectory analysis
│   └── utils.py                # Plotting utilities
│
├── output/                       # Final output figures
│
└── docs/                         # Sphinx documentation
    ├── conf.py
    ├── index.rst
    └── _build/
```

---

## Workflow

### Typical Usage Workflow

#### 1. Prepare Abstraction Data
```bash
# Generate nominal abstraction (conservative)
python RunFileAbstraction.py --model Integrators --epsilons '[0.1]' \
    --simulation_id '01' --store_abstraction_data 'True'

# Generate robust abstraction (with uncertainty margins)
python RunFileAbstraction.py --model Integrators --epsilons '[0.05]' \
    --simulation_id '02' --store_abstraction_data 'True'
```

#### 2. Configure MPC Parameters
Edit `RunFileMPC.py` or use command-line arguments to set:
- Cost weights (--state_weight, --input_weight)
- Prediction horizon (--prediction_horizon)
- Noise parameters (--cov_noise, --cov_initial_state)

#### 3. Run MPC Simulation
```bash
python RunFileMPC.py \
    --model 'Double_integrator' \
    --abstraction_data_nominal 'abstraction_data_DoubleIntegrator_01' \
    --abstraction_data 'abstraction_data_DoubleIntegrator_02' \
    --number_experiments 10 \
    --store_simulation_data 'True' \
    --plot_simulation 'True'
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

### Step 2: Design PWA Approximation (Critical!)
Create `mpc_core/YourModel_dynamics.py`:
- Partition state space into regions
- Fit affine approximations in each region
- Implement PWA matrices (A, B, g)
- Follow examples: Double_integrator, mountain_car, Dubins_small

### Step 3: Build MPC Optimization Model
Add to `mpc_core/optimization_model_builders.py`:
- `optimization_model_builder_yourmodel()` function
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

## Performance Tuning Guide

### Improving Solution Speed

1. **Increase MIPGap** in `optimization_model_builders.py`
   ```python
   MPC_model.Params.MIPGap = 0.10  # 10% gap instead of 5%
   ```

2. **Reduce Prediction Horizon**
   ```bash
   --prediction_horizon 2  # Instead of 3-5
   ```

3. **Lower PWA Region Count**
   ```bash
   --number_pwa_regions 50  # Coarser approximation
   ```

4. **Enable Parallel Processing**
   ```python
   MPC_model.Params.Threads = 0  # Use all cores
   ```

### Improving Control Performance

1. **Increase Prediction Horizon**
   ```bash
   --prediction_horizon 5  # More foresight
   ```

2. **Decrease MIPGap**
   ```python
   MPC_model.Params.MIPGap = 0.01  # 1% gap
   ```

3. **Increase PWA Region Count**
   ```bash
   --number_pwa_regions 200  # Better approximation
   ```

4. **Adjust Cost Weights**
   - Increase `--state_weight` to penalize state deviations more
   - Adjust `--input_weight` for control effort trade-off

---

## References

This work combines:
- **IMDP Abstraction**: Formal abstraction of stochastic systems into Markov models
- **Mixed Integer Programming**: Optimization of hybrid systems via Gurobi
- **Piecewise Affine Systems**: Hybrid approximations of nonlinear dynamics
- **Model Predictive Control**: Receding horizon optimization for feedback control

See [ReadMeAbstraction.md](ReadMeAbstraction.md) for original IMDP abstraction documentation.

---

## Support and Troubleshooting

### Common Issues

**Issue**: Gurobi "license not found" error
- **Solution**: Activate Gurobi academic license from [Gurobi website](https://www.gurobi.com/academia/academic-license-access/)

**Issue**: Memory errors during abstraction generation
- **Solution**: Reduce batch size in `RunFileAbstraction.py` or use GPU acceleration

**Issue**: MPC solver timeout or infeasible
- **Solution**: Check abstraction data matches model, reduce prediction horizon, adjust cost weights

**Issue**: PWA approximation quality poor
- **Solution**: Increase number of PWA regions, verify affine approximations in code

---

## Citation

If you use this framework in your research, please cite:
- Original IMDP abstraction work (see ReadMeAbstraction.md)
- This extension combining abstraction with MPC

---

## License

[Specify your license here]

---

**Last Updated**: March 2026
