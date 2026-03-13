import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB


def optimization_model_builder_double_integrator(PREDICTION_HORIZON, STATES_NUMBER, INPUTS_NUMBER, lower_bound_x, upper_bound_x, upper_bound_u, lower_bound_u, NUMBER_PWA_REGIONS, centers, M_state, m_state, lb_z_s_k, ub_z_s_k, M_input_policy, m_input_policy):
    print(f"Start general MPC model contruction")
    start_time = time.perf_counter()

    MPC_model = gp.Model("MPC")

    Ns = centers.shape[0]

    # Optimization variables
    ub_u_matrix = np.zeros((PREDICTION_HORIZON,INPUTS_NUMBER))
    lb_u_matrix = np.zeros((PREDICTION_HORIZON,INPUTS_NUMBER))

    ub_x_matrix = np.zeros((PREDICTION_HORIZON+1,STATES_NUMBER))
    lb_x_matrix = np.zeros((PREDICTION_HORIZON+1,STATES_NUMBER))

    for i in range(PREDICTION_HORIZON):
        ub_u_matrix[i,:] = upper_bound_u 
        lb_u_matrix[i,:] = lower_bound_u

        ub_x_matrix[i,:] = upper_bound_x
        lb_x_matrix[i,:] = lower_bound_x

    ub_x_matrix[-1,:] = upper_bound_x
    lb_x_matrix[-1,:] = lower_bound_x

    u_tilde_k = MPC_model.addMVar((PREDICTION_HORIZON,INPUTS_NUMBER), ub=ub_u_matrix, lb=lb_u_matrix)
    x_tilde_k = MPC_model.addMVar((PREDICTION_HORIZON+1,STATES_NUMBER), ub=ub_x_matrix, lb=lb_x_matrix)

    # lb_z_tilde_1 = MIN_ANGLE*np.ones((PREDICTION_HORIZON,NUMBER_PWA_REGIONS-1))
    # ub_z_tilde_1 = MAX_ANGLE*np.ones((PREDICTION_HORIZON,NUMBER_PWA_REGIONS-1))

    # delta_tilde_1_k = MPC_model.addMVar((PREDICTION_HORIZON,NUMBER_PWA_REGIONS-1), vtype=GRB.BINARY)
    # z_tilde_1_k = MPC_model.addMVar((PREDICTION_HORIZON,NUMBER_PWA_REGIONS-1), lb=lb_z_tilde_1, ub=ub_z_tilde_1)

    # Optimization reconstruction variables 
    delta_s_k = MPC_model.addMVar((PREDICTION_HORIZON+1,Ns), vtype=GRB.BINARY)
    z_s_k = MPC_model.addMVar((PREDICTION_HORIZON+1,STATES_NUMBER,Ns), lb=lb_z_s_k, ub=ub_z_s_k)


    # lb_w = V_MIN*np.ones(PREDICTION_HORIZON)
    # ub_w = V_MAX*np.ones(PREDICTION_HORIZON)

    # w_1_k = MPC_model.addMVar(PREDICTION_HORIZON, lb=lb_w, ub=ub_w)
    # w_2_k = MPC_model.addMVar(PREDICTION_HORIZON, lb=lb_w, ub=ub_w)


    # State abstraction constraints for initial state

    EPSILON = 1e-6
    # EPSILON = 0

    for cell_idx in range(Ns):
        MPC_model.addConstr(z_s_k[0,:,cell_idx] <= (M_state[cell_idx,:]-EPSILON)*delta_s_k[0,cell_idx])
        MPC_model.addConstr(z_s_k[0,:,cell_idx] >= (m_state[cell_idx,:]+EPSILON)*delta_s_k[0,cell_idx])

        MPC_model.addConstr(z_s_k[0,:,cell_idx] <= x_tilde_k[0,:] - lower_bound_x*(1 - delta_s_k[0,cell_idx]))
        MPC_model.addConstr(z_s_k[0,:,cell_idx] >= x_tilde_k[0,:] - upper_bound_x*(1 - delta_s_k[0,cell_idx]))
    # Cost definition
    cost = 0



    # Building prediction model
    for j in range(PREDICTION_HORIZON):

        # Input Constraint
        sum_upper_bounds = 0
        sum_lower_bounds = 0
        for cell_idx in range(Ns):
            sum_upper_bounds += M_input_policy[cell_idx,:]*delta_s_k[j,cell_idx]
            sum_lower_bounds += m_input_policy[cell_idx,:]*delta_s_k[j,cell_idx]

        
        MPC_model.addConstr(u_tilde_k[j,:] <= sum_upper_bounds)
        MPC_model.addConstr(u_tilde_k[j,:] >= sum_lower_bounds)

        # State abstraction constraints
        for cell_idx in range(Ns):
            MPC_model.addConstr(z_s_k[j+1,:,cell_idx] <= (M_state[cell_idx,:] - EPSILON)*delta_s_k[j+1,cell_idx])
            MPC_model.addConstr(z_s_k[j+1,:,cell_idx] >= (m_state[cell_idx,:] + EPSILON)*delta_s_k[j+1,cell_idx])

            MPC_model.addConstr(z_s_k[j+1,:,cell_idx] <= x_tilde_k[j+1,:] - lower_bound_x*(1 - delta_s_k[j+1,cell_idx]))
            MPC_model.addConstr(z_s_k[j+1,:,cell_idx] >= x_tilde_k[j+1,:] - upper_bound_x*(1 - delta_s_k[j+1,cell_idx]))

        # Abstraciton consistency (first true by assumption)
        delta_s_k_sum = 0

        for cell_idx in range(Ns):
            delta_s_k_sum += delta_s_k[j+1,cell_idx]

        # constraints += [delta_s_k_sum == 1]
        # MPC_model.addConstr(delta_s_k_sum == 1)
        MPC_model.addConstr(delta_s_k_sum <= 1)
        

        tau = 1.0

        # State transition matrix
        A  = np.array([[1, tau],
                        [0, 1]])
        
        # Input matrix
        B  = np.array([[tau**2/2],
                        [tau]])

        # Prediction step
        MPC_model.addConstr(x_tilde_k[j+1,:] == A @ x_tilde_k[j,:] + B @ u_tilde_k[j,:])

    # Optimization options
    MPC_model.Params.OutputFlag = 0             # print solver log (1 = on, 0 = off)
    MPC_model.Params.LogToConsole = 0
    MPC_model.Params.MIPFocus    = 3             # 1=feas, 2=bound, 3=optimality
    MPC_model.Params.MIPGap    = 0.05            # 5% gap 
    # MPC_model.Params.MIPGap    = 0.10            # 10% gap  
    # MPC_model.Params.MIPGap    = 0.001            # 0.1% gap  
    MPC_model.Params.Heuristics  = 0.25          # 0.1–0.5
    MPC_model.Params.Threads     = 0                    # 0 = all cores

    MPC_model.update()
    elapsed_time = time.perf_counter() - start_time
    print(f"Model construction required: {elapsed_time:.6f} seconds")

    return MPC_model

def optimization_model_builder_mountain_car():
    return None

def optimization_model_builder_small_dubins(PREDICTION_HORIZON, STATES_NUMBER, INPUTS_NUMBER, lower_bound_x, upper_bound_x, upper_bound_u, lower_bound_u, NUMBER_PWA_REGIONS, centers, M_state, m_state, lb_z_s_k, ub_z_s_k, M_input_policy, m_input_policy):
    print(f"Start general MPC model contruction")
    start_time = time.perf_counter()

    MPC_model = gp.Model("MPC")

    angles = np.linspace(-np.pi,np.pi,NUMBER_PWA_REGIONS)

    V_MIN = lower_bound_u[1]
    V_MAX = upper_bound_u[1]

    FUN_MIN = -1
    FUN_MAX = 1

    MIN_ANGLE = lower_bound_x[2]
    MAX_ANGLE = upper_bound_x[2]

    Ns = centers.shape[0]

    tau = 1
    alpha = 0.85

    # Optimization variables
    ub_u_matrix = np.zeros((PREDICTION_HORIZON,INPUTS_NUMBER))
    lb_u_matrix = np.zeros((PREDICTION_HORIZON,INPUTS_NUMBER))

    ub_x_matrix = np.zeros((PREDICTION_HORIZON+1,STATES_NUMBER))
    lb_x_matrix = np.zeros((PREDICTION_HORIZON+1,STATES_NUMBER))

    for i in range(PREDICTION_HORIZON):
        ub_u_matrix[i,:] = upper_bound_u 
        lb_u_matrix[i,:] = lower_bound_u

        ub_x_matrix[i,:] = upper_bound_x
        lb_x_matrix[i,:] = lower_bound_x

    ub_x_matrix[-1,:] = upper_bound_x
    lb_x_matrix[-1,:] = lower_bound_x

    u_tilde_k = MPC_model.addMVar((PREDICTION_HORIZON,INPUTS_NUMBER), ub=ub_u_matrix, lb=lb_u_matrix)
    x_tilde_k = MPC_model.addMVar((PREDICTION_HORIZON+1,STATES_NUMBER), ub=ub_x_matrix, lb=lb_x_matrix)

    lb_z_tilde_1 = MIN_ANGLE*np.ones((PREDICTION_HORIZON,NUMBER_PWA_REGIONS-1))
    ub_z_tilde_1 = MAX_ANGLE*np.ones((PREDICTION_HORIZON,NUMBER_PWA_REGIONS-1))

    delta_tilde_1_k = MPC_model.addMVar((PREDICTION_HORIZON,NUMBER_PWA_REGIONS-1), vtype=GRB.BINARY)
    z_tilde_1_k = MPC_model.addMVar((PREDICTION_HORIZON,NUMBER_PWA_REGIONS-1), lb=lb_z_tilde_1, ub=ub_z_tilde_1)

    # [-pi,pi]
    lb_z_tilde_2 = 2*MIN_ANGLE*np.ones(PREDICTION_HORIZON)
    ub_z_tilde_2 = 2*MAX_ANGLE*np.ones(PREDICTION_HORIZON)
    delta_tilde_2_k = MPC_model.addMVar(PREDICTION_HORIZON, vtype=GRB.BINARY)
    z_tilde_2_k = MPC_model.addMVar(PREDICTION_HORIZON, lb=lb_z_tilde_2, ub=ub_z_tilde_2)

    # [pi,2pi]
    lb_z_tilde_3 = 2*MIN_ANGLE*np.ones(PREDICTION_HORIZON)
    ub_z_tilde_3 = 2*MAX_ANGLE*np.ones(PREDICTION_HORIZON)
    delta_tilde_3_k = MPC_model.addMVar(PREDICTION_HORIZON, vtype=GRB.BINARY)
    z_tilde_3_k = MPC_model.addMVar(PREDICTION_HORIZON, lb=lb_z_tilde_3, ub=ub_z_tilde_3)

    # [-2pi,-pi]
    lb_z_tilde_4 = 2*MIN_ANGLE*np.ones(PREDICTION_HORIZON)
    ub_z_tilde_4 = 2*MAX_ANGLE*np.ones(PREDICTION_HORIZON)
    delta_tilde_4_k = MPC_model.addMVar(PREDICTION_HORIZON, vtype=GRB.BINARY)
    z_tilde_4_k = MPC_model.addMVar(PREDICTION_HORIZON, lb=lb_z_tilde_4, ub=ub_z_tilde_4)

    # Optimization reconstruction variables 
    delta_s_k = MPC_model.addMVar((PREDICTION_HORIZON+1,Ns), vtype=GRB.BINARY)
    z_s_k = MPC_model.addMVar((PREDICTION_HORIZON+1,STATES_NUMBER,Ns), lb=lb_z_s_k, ub=ub_z_s_k)

    # Optimization reconstruction variables 
    delta_s_k = MPC_model.addMVar((PREDICTION_HORIZON+1,Ns), vtype=GRB.BINARY)
    z_s_k = MPC_model.addMVar((PREDICTION_HORIZON+1,3,Ns), lb=lb_z_s_k, ub=ub_z_s_k)


    lb_w = V_MIN*np.ones(PREDICTION_HORIZON)
    ub_w = V_MAX*np.ones(PREDICTION_HORIZON)

    w_1_k = MPC_model.addMVar(PREDICTION_HORIZON, lb=lb_w, ub=ub_w)
    w_2_k = MPC_model.addMVar(PREDICTION_HORIZON, lb=lb_w, ub=ub_w)


    # State abstraction constraints for initial state

    EPSILON = 1e-6
    # EPSILON = 0

    for cell_idx in range(Ns):
        MPC_model.addConstr(z_s_k[0,:,cell_idx] <= (M_state[cell_idx,:]-EPSILON)*delta_s_k[0,cell_idx])
        MPC_model.addConstr(z_s_k[0,:,cell_idx] >= (m_state[cell_idx,:]+EPSILON)*delta_s_k[0,cell_idx])

        MPC_model.addConstr(z_s_k[0,:,cell_idx] <= x_tilde_k[0,:] - lower_bound_x*(1 - delta_s_k[0,cell_idx]))
        MPC_model.addConstr(z_s_k[0,:,cell_idx] >= x_tilde_k[0,:] - upper_bound_x*(1 - delta_s_k[0,cell_idx]))
    # Cost definition
    cost = 0



    # Building prediction model
    for j in range(PREDICTION_HORIZON):

        # Input Constraint
        sum_upper_bounds = 0
        sum_lower_bounds = 0
        for cell_idx in range(Ns):
            sum_upper_bounds += M_input_policy[cell_idx,:]*delta_s_k[j,cell_idx]
            sum_lower_bounds += m_input_policy[cell_idx,:]*delta_s_k[j,cell_idx]

        
        MPC_model.addConstr(u_tilde_k[j,:] <= sum_upper_bounds)
        MPC_model.addConstr(u_tilde_k[j,:] >= sum_lower_bounds)

        # State abstraction constraints
        for cell_idx in range(Ns):
            MPC_model.addConstr(z_s_k[j+1,:,cell_idx] <= (M_state[cell_idx,:] - EPSILON)*delta_s_k[j+1,cell_idx])
            MPC_model.addConstr(z_s_k[j+1,:,cell_idx] >= (m_state[cell_idx,:] + EPSILON)*delta_s_k[j+1,cell_idx])

            MPC_model.addConstr(z_s_k[j+1,:,cell_idx] <= x_tilde_k[j+1,:] - lower_bound_x*(1 - delta_s_k[j+1,cell_idx]))
            MPC_model.addConstr(z_s_k[j+1,:,cell_idx] >= x_tilde_k[j+1,:] - upper_bound_x*(1 - delta_s_k[j+1,cell_idx]))

        # Abstraciton consistency (first true by assumption)
        delta_s_k_sum = 0

        for cell_idx in range(Ns):
            delta_s_k_sum += delta_s_k[j+1,cell_idx]

        # constraints += [delta_s_k_sum == 1]
        # MPC_model.addConstr(delta_s_k_sum == 1)
        MPC_model.addConstr(delta_s_k_sum <= 1)
        

        #  Linear approximaiton
        sin_theta_approximation = 0
        cos_theta_approximation = 0

        V_cos_approximation = 0 
        V_sin_approximation = 0

        delta_1_sum = 0


        for i in range(NUMBER_PWA_REGIONS-1):
            
            delta_1_sum += delta_tilde_1_k[j,i]
            

            MPC_model.addConstr(x_tilde_k[j,2] >= (angles[i]+EPSILON)*delta_tilde_1_k[j,i] + MIN_ANGLE*(1 - delta_tilde_1_k[j,i]))
            MPC_model.addConstr(x_tilde_k[j,2] <= (angles[i+1]-EPSILON)*delta_tilde_1_k[j,i] + MAX_ANGLE*(1 - delta_tilde_1_k[j,i]))

            MPC_model.addConstr(z_tilde_1_k[j,i] >= MIN_ANGLE*delta_tilde_1_k[j,i])
            MPC_model.addConstr(z_tilde_1_k[j,i] <= MAX_ANGLE*delta_tilde_1_k[j,i])
            MPC_model.addConstr(z_tilde_1_k[j,i] <= x_tilde_k[j,2] - MIN_ANGLE*(1-delta_tilde_1_k[j,i]))
            MPC_model.addConstr(z_tilde_1_k[j,i] >= x_tilde_k[j,2] - MAX_ANGLE*(1-delta_tilde_1_k[j,i]))

            a_i_sin = (np.sin(angles[i+1])- np.sin(angles[i]))/(angles[i+1] - angles[i])
            b_i_sin = np.sin(angles[i]) - a_i_sin*angles[i]

            sin_theta_approximation += a_i_sin*z_tilde_1_k[j,i] + b_i_sin*delta_tilde_1_k[j,i]

            a_i_cos = (np.cos(angles[i+1])- np.cos(angles[i]))/(angles[i+1] - angles[i])
            b_i_cos = np.cos(angles[i]) - a_i_cos*angles[i]

            cos_theta_approximation += a_i_cos*z_tilde_1_k[j,i] + b_i_cos*delta_tilde_1_k[j,i]

        # Linear approximation consistency 
        MPC_model.addConstr(delta_1_sum == 1)
        

        # McCormick-envelope inequalities
        MPC_model.addConstr(w_1_k[j] >= V_MIN*sin_theta_approximation + FUN_MIN*u_tilde_k[j,1] - V_MIN*FUN_MIN)
        MPC_model.addConstr(w_1_k[j] >= V_MAX*sin_theta_approximation + FUN_MAX*u_tilde_k[j,1] - V_MAX*FUN_MAX)
        MPC_model.addConstr(w_1_k[j] <= V_MIN*sin_theta_approximation + FUN_MAX*u_tilde_k[j,1] - V_MIN*FUN_MAX)
        MPC_model.addConstr(w_1_k[j] <= V_MAX*sin_theta_approximation + FUN_MIN*u_tilde_k[j,1] - V_MAX*FUN_MIN)

        MPC_model.addConstr(w_2_k[j] >= V_MIN*cos_theta_approximation + FUN_MIN*u_tilde_k[j,1] - V_MIN*FUN_MIN)
        MPC_model.addConstr(w_2_k[j] >= V_MAX*cos_theta_approximation + FUN_MAX*u_tilde_k[j,1] - V_MAX*FUN_MAX)
        MPC_model.addConstr(w_2_k[j] <= V_MIN*cos_theta_approximation + FUN_MAX*u_tilde_k[j,1] - V_MIN*FUN_MAX)
        MPC_model.addConstr(w_2_k[j] <= V_MAX*cos_theta_approximation + FUN_MIN*u_tilde_k[j,1] - V_MAX*FUN_MIN)

        theta_plus = x_tilde_k[j,2] + tau * alpha * u_tilde_k[j,0]

        #Constraints [-pi,pi]
        MPC_model.addConstr(z_tilde_2_k[j] >= MIN_ANGLE*delta_tilde_2_k[j])
        MPC_model.addConstr(z_tilde_2_k[j] <= MAX_ANGLE*delta_tilde_2_k[j])
        MPC_model.addConstr(z_tilde_2_k[j] <= theta_plus - 2*MIN_ANGLE*(1-delta_tilde_2_k[j]))
        MPC_model.addConstr(z_tilde_2_k[j] >= theta_plus - 2*MAX_ANGLE*(1-delta_tilde_2_k[j]))

        #Constraints [pi,2pi]
        MPC_model.addConstr(z_tilde_3_k[j] >= (MAX_ANGLE+EPSILON)*delta_tilde_3_k[j])
        MPC_model.addConstr(z_tilde_3_k[j] <= 2*MAX_ANGLE*delta_tilde_3_k[j])
        MPC_model.addConstr(z_tilde_3_k[j] <= theta_plus - 2*MIN_ANGLE*(1-delta_tilde_3_k[j]))
        MPC_model.addConstr(z_tilde_3_k[j] >= theta_plus - 2*MAX_ANGLE*(1-delta_tilde_3_k[j]))

        #Constraints [-2pi,-pi]
        MPC_model.addConstr(z_tilde_4_k[j] >= 2*MIN_ANGLE*delta_tilde_4_k[j])
        MPC_model.addConstr(z_tilde_4_k[j] <= (MIN_ANGLE-EPSILON)*delta_tilde_4_k[j])
        MPC_model.addConstr(z_tilde_4_k[j] <= theta_plus - 2*MIN_ANGLE*(1-delta_tilde_4_k[j]))
        MPC_model.addConstr(z_tilde_4_k[j] >= theta_plus - 2*MAX_ANGLE*(1-delta_tilde_4_k[j]))

        # Prediction step
        MPC_model.addConstr(x_tilde_k[j+1,0] == x_tilde_k[j,0] + tau * w_2_k[j])
        MPC_model.addConstr(x_tilde_k[j+1,1] == x_tilde_k[j,1] + tau * w_1_k[j])
        MPC_model.addConstr(x_tilde_k[j+1,2] == theta_plus + delta_tilde_3_k[j]*(-2*np.pi) + delta_tilde_4_k[j]*(2*np.pi))

    # Optimization options
    MPC_model.Params.OutputFlag = 0             # print solver log (1 = on, 0 = off)
    MPC_model.Params.LogToConsole = 0
    MPC_model.Params.MIPFocus    = 1             # 1=feas, 2=bound, 3=optimality
    # MPC_model.Params.MIPGap    = 0.05            # 5% gap 
    MPC_model.Params.MIPGap    = 0.10            # 10% gap 
    # MPC_model.Params.MIPGap    = 0.25            # 10% gap  
    # MPC_model.Params.MIPGap    = 0.001            # 0.1% gap  
    MPC_model.Params.Heuristics  = 0.25          # 0.1–0.5
    MPC_model.Params.Threads     = 0                    # 0 = all cores

    MPC_model.update()
    elapsed_time = time.perf_counter() - start_time
    print(f"Model construction required: {elapsed_time:.6f} seconds")

    return MPC_model