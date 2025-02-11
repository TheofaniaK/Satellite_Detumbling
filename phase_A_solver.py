from acados_template import AcadosOcp, AcadosOcpSolver
from scipy.linalg import block_diag
import numpy as np
from phase_A_model import phase_A_model, tau_computation
from casadi import vertcat, DM

def phase_A_solver():
    ocp = AcadosOcp()

    model = phase_A_model()

    # Set the model
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu

    x = model.x
    u = model.u

    # Number of shooting intervals
    N_horizon = 50
    ocp.dims.N = N_horizon

    # Cost function
    ocp.cost.cost_type = 'NONLINEAR_LS'   # 'CONVEX_OVER_NONLINEAR'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    ocp.model.cost_y_expr = vertcat(x, u)  # States and controls
    ocp.model.cost_y_expr_e = x  # Terminal cost on states

    # State cost matrix (penalizing deviations in states)
    Q = 400*block_diag(2, 2, 0.001,
                   7, 7, 9,
                   9, 9, 12, 15) # state: [r_b, omega_b, q]

    # Control cost matrix (penalizing control effort)
    #R = 1*block_diag(0.4, 0.8, 0.5)
    R = 2 * block_diag(0.8, 0.4, 0.6)

    # Set the weight matrices
    ocp.cost.W = block_diag(Q, R)
    ocp.cost.W_e = Q  # Terminal cost on the states

    # Mapping for cost terms
    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)  # Map state to cost

    Vu = np.zeros((ny, nu))
    Vu[nx:, :] = np.eye(nu)  # Map control to cost
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)  # Terminal state mapping

    theta_ref = [0.05, 0.4, 0.05]
    theta_dot_ref = [0.0, 0.0, 0.0]
    omega_b_ref = [0.0, 0.0, 0.2]
    tau_ref = tau_computation(omega_b_ref, theta_ref, theta_dot_ref)
    tau_ref = np.reshape(DM(tau_ref), (1, -1))

    # Reference trajectory
    desired_x = np.array([0.0, 0.0, 10,
                          0.0, 0.0, 0.2,
                          0.0, 0.0, 0.0, 1.0])  # Target state

    yref = np.hstack((desired_x, tau_ref[0]))  # State + control reference
    ocp.cost.yref = yref
    ocp.cost.yref_e = desired_x

    # State constraints
    x_min = 1*np.array([-2, -2, -1,
                      -0.3, -0.3, -0.1,
                      -0.2, -0.6, -1.2, -1.0])  # min state bounds
    x_max = 1*np.array([4.0, 4.0, 2000,
                      0.2, 0.3, 0.4,
                      0.9, 0.9, 0.9, 1.25])  # max state bounds
    ocp.constraints.lbx = x_min
    ocp.constraints.ubx = x_max
    ocp.constraints.idxbx = np.arange(nx) # map x(t) onto its bound vectors

    # Define terminal constraints
    epsilon = 2 * np.ones(len(x_min))
    epsilon[2] = 2000
    # ocp.constraints.constr_type_e = 'BGH'       # Constraint type at terminal stage
    ocp.constraints.lbx_e = desired_x - epsilon
    ocp.constraints.ubx_e = desired_x + epsilon
    ocp.constraints.idxbx_e = np.arange(nx)  # Constrain all states at terminal stage

    # Control input constraints
    u_min = 2*np.array([-1, -1, -1])  # large min control bounds
    u_max = 2*np.array([1, 1, 1])  # large max control bounds
    ocp.constraints.lbu = u_min
    ocp.constraints.ubu = u_max
    ocp.constraints.idxbu = np.arange(nu) # map u(t) onto its bound vectors

    # Initial state
    initial_state = np.array([0.0, 0.0, 0.0,
                              0.1, 0.05, 0.01,
                              0.5, 0.2, 0.4, -0.8]) # Example initial state
    ocp.constraints.x0 = initial_state

    # Solver options
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'  # turns on globalization
    ocp.solver_options.regularize_method = 'MIRROR'

    ocp.solver_options.nlp_solver_type = 'SQP'  # Sequential Quadratic Programming
    ocp.solver_options.nlp_solver_max_iter = 150
    # ocp.solver_options.nlp_solver_tol_eq = 1e-9

    ocp.solver_options.integrator_type = 'IRK'  # Implicit Runge-Kutta
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 2

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  # Quadratic Program solver
    # ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.qp_solver_cond_N = N_horizon
    # ocp.solver_options.qp_solver_tol = 1e-6
    # ocp.solver_options.qp_tol = 1e-5

    ocp.solver_options.tol = 1e-5
    ocp.solver_options.N_horizon = N_horizon  # number of shooting intervals
    # ocp.dims.N = 8

    # set prediction horizon
    ocp.solver_options.tf = .350

    # Create the solver
    ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp_' + model.name + '.json')

    return ocp_solver
