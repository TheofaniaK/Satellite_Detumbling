from acados_template import AcadosModel
from casadi import SX, inv, vertcat, horzcat, blockcat, sin, cos, solve, mtimes
import numpy as np

################################## Functions ###################################
def quaternion_multiplication(q1, q2):
    """Quaternion multiplication for two quaternions q1 and q2"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return SX([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ])

def quaternion_conjugate(q):
    """Returns the conjugate (inverse for unit quaternion) of quaternion q"""
    return horzcat(-q[0], -q[1], -q[2], q[3])

def quaternion_error(q, q_desired):
    """Calculate quaternion error q_error = q_desired * q^-1"""
    q_inv = quaternion_conjugate(q)
    return quaternion_multiplication(q_desired, q_inv)

def quaternion_to_rotation_matrix(q):
    """Rotation matrix A(q) from quaternion q"""
    q_v = q[:3]  # Vector part of the quaternion q
    q_o = q[3]  # Scalar part of the quaternion q

    I = SX.eye(3)
    q_v_cross = skew_symmetric(q_v)
    return I + 2 * q_o * q_v_cross + 2 * (q_v_cross @ q_v_cross)

def relative_angular_velocity(omega_b, omega_s, q):
    A_q = quaternion_to_rotation_matrix(q)
    return omega_b - A_q @ omega_s

def Omega(omega_rel):
    """Construct the Omega matrix for quaternion dynamics"""
    omega_skew = skew_symmetric(omega_rel)
    omega_vector = omega_rel
    omega_transpose = omega_rel.T

    # Construct the Omega matrix
    return vertcat(horzcat(-omega_skew, omega_vector),
                   horzcat(-omega_transpose, SX.zeros((1, 1))))

def parallel_axis_theorem(I_cm, m, r, dr_dt, dr_dtheta):
    """Calculate the inertia tensor about a new axis using the parallel axis theorem"""
    # Calculate r·r (dot product), resulting in a scalar
    r_dot_r = mtimes(r.T, r)  # Scalar value
    dr_dot_r_dt = mtimes(dr_dt.T, r) + mtimes(r.T, dr_dt)

    # Identity matrix (3x3)
    I_3 = np.eye(3)

    # Calculate the outer product r ⊗ r (3x3 matrix)
    r_outer_r = mtimes(r, r.T)  # 3x3 matrix
    dr_outer_r_dt = mtimes(dr_dt, r.T) + mtimes(r, dr_dt.T)

    # Apply the parallel axis theorem
    I_A = I_cm + m * (r_dot_r * I_3 - r_outer_r)
    dI_A_dt = m * (dr_dot_r_dt * I_3 - dr_outer_r_dt)

    # Derivative wrt theta1
    d_I_A_dtheta1 = m * ((mtimes(r.T, dr_dtheta[:,0]) + mtimes(dr_dtheta[:,0].T, r)) * SX.eye(3) -
                         mtimes(dr_dtheta[:,0],r.T) - mtimes(r,dr_dtheta[:,0].T))
    # Derivative wrt theta2
    d_I_A_dtheta2 = m * ((mtimes(r.T, dr_dtheta[:,1]) + mtimes(dr_dtheta[:,1].T, r)) * SX.eye(3) -
                         mtimes(dr_dtheta[:,1],r.T) - mtimes(r,dr_dtheta[:,1].T))
    # Derivative wrt theta3
    d_I_A_dtheta3 = m * ((mtimes(r.T, dr_dtheta[:,2]) + mtimes(dr_dtheta[:,2].T, r)) * SX.eye(3) -
                         mtimes(dr_dtheta[:,2],r.T) - mtimes(r,dr_dtheta[:,2].T))

    return I_A, dI_A_dt, [d_I_A_dtheta1, d_I_A_dtheta2, d_I_A_dtheta3]

def skew_symmetric(v):
    """Assume a vector v = [v1, v2, v3]"""
    return vertcat(
        horzcat(0, -v[2], v[1]),
        horzcat(v[2], 0, -v[0]),
        horzcat(-v[1], v[0], 0)
    )

def RW_inertia_matrix(m, r_1, r_2, h, r_RW_x, r_RW_y, r_RW_z):
    """
    Calculates the inertia matrices for 3 reaction wheels using the given formulae.
    The function returns the inertia matrices for RW1, RW2, and RW3.
    """
    Ι1 = 0.5 * m * (r_2 ** 2 + r_1 ** 2)
    Ι2 = (1 / 12) * m * (3 * (r_2 ** 2 + r_1 ** 2) + 4 * h ** 2)

    # Define the inertia matrix for RW1 - around x-axis
    I_RW1 = SX([
        [Ι1, 0, 0],
        [0, Ι2, 0],
        [0, 0, Ι2]
    ])

    # Define the inertia matrix for RW2 - around y-axis
    I_RW2 = SX([
        [Ι2, 0, 0],
        [0, Ι1, 0],
        [0, 0, Ι2]
    ])

    # Define the inertia matrix for RW3 - around z-axis
    I_RW3 = SX([
        [Ι2, 0, 0],
        [0, Ι2, 0],
        [0, 0, Ι1]
    ])

    # RW inertia matrices in base frame
    I_RW1_base, _, _ = parallel_axis_theorem(I_RW1, m, SX([r_RW_x, 0, 0]), SX.zeros((3, 1)), SX.zeros((3, 3)))
    I_RW2_base, _, _ = parallel_axis_theorem(I_RW2, m, SX([0, r_RW_y, 0]), SX.zeros((3, 1)), SX.zeros((3, 3)))
    I_RW3_base, _, _ = parallel_axis_theorem(I_RW3, m, SX([0, 0, r_RW_z]), SX.zeros((3, 1)), SX.zeros((3, 3)))

    return I_RW1_base, I_RW2_base, I_RW3_base


def rotate_inertia_tensor(I_CM, axis=None, theta_deg=None):
    """
    Rotates the inertia tensor of a cylinder by a specified angle around a specified axis.
    """
    theta_rad = np.radians(theta_deg)  # Convert angle to radians

    # Define rotation matrix based on axis
    if axis == 'x':
        R = SX([
            [1, 0, 0],
            [0, cos(theta_rad), -sin(theta_rad)],
            [0, sin(theta_rad), cos(theta_rad)]
        ])
    elif axis == 'y':
        R = SX([
            [cos(theta_rad), 0, sin(theta_rad)],
            [0, 1, 0],
            [-sin(theta_rad), 0, cos(theta_rad)]
        ])
    elif axis == 'z':
        R = SX([
            [cos(theta_rad), -sin(theta_rad), 0],
            [sin(theta_rad), cos(theta_rad), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis: Use 'x', 'y', or 'z'.")

    # Apply the rotation: I_new = R * I_CM * R^T
    I_new = R @ I_CM @ R.T
    return I_new

def joint_inertia_tensor(m, r, L):
    """
    Returns the inertia tensor for a cylinder rotating about its principal axis.
    """
    return SX([
        [1 / 12 * m * (3 * r ** 2 + L ** 2), 0, 0],
        [0, 1 / 12 * m * (3 * r ** 2 + L ** 2), 0],
        [0, 0, 1 / 2 * m * r ** 2]
    ])

def quaternion_dynamics(t, q, omega_b, omega_s):
    """Define the dynamics for quaternion integration"""
    omega_rel = relative_angular_velocity(omega_b, omega_s, q)
    Omega_matrix = Omega(omega_rel)  # Construct the Omega matrix
    dot_q = 0.5 * Omega_matrix @ q  # Update quaternion dynamics

    return dot_q


def export_SA_model():
    model_name = "Sem_proj"
    ns = 6 + 4      # number of states
    nsd = 6         # number of states dot
    nx = ns + nsd   # total number of states
    nu = 6          # number of control input

    # State Variables
    r_b = SX.sym('r_b', 3, 1)                            # angular displacement
    r_b_dot = SX.sym('omega_b', 3, 1)                    # angular velocity
    theta = SX.sym('theta', 3, 1)                        # joint angles
    theta_dot = SX.sym('theta_dot', 3, 1)                # joint angular acceleration
    q = SX.sym('q', 4, 1)                                # quaternion
    x = vertcat(r_b, theta, r_b_dot, theta_dot, q) # state vector
    xdot = SX.sym('xdot', nx, 1)                         # state derivative

    # Control Input
    tau = SX.sym('tau', nu, 1)                           # torques

        ############################### Building Dynamics ###################################

    w = vertcat(r_b, theta)
    dw = vertcat(r_b_dot, theta_dot)
    H, dH_dt, dH_dtheta1, dH_dtheta2, dH_dtheta3, m_c = components(w, dw)


    # Extract relevant sub-matrices from H and dH_dt
    H_V = H[0][0]
    H_V_omega = H[0][1]
    H_V_theta = H[0][2]
    H_omega = H[1][1]
    H_omega_theta = H[1][2]
    H_omega_r = H[1][3]
    H_theta = H[2][2]
    H_theta_r = H[2][3]
    H_r = H[3][3]

    dH_V_dt = dH_dt[0][0]
    dH_V_omega_dt = dH_dt[0][1]
    dH_V_theta_dt = dH_dt[0][2]
    dH_V_r_dt = dH_dt[0][3]

    dH_omega_dt = dH_dt[1][1]
    dH_omega_theta_dt = dH_dt[1][2]
    dH_omega_r_dt = dH_dt[1][3]

    dH_theta_dt = dH_dt[2][2]
    dH_theta_r_dt = dH_dt[2][3]
    dH_r_dt = dH_dt[3][3]

    # Inverse of H_V for use later
    H_V_inv = inv(H_V)

    M_b = H_omega - H_V_omega.T @ H_V_inv @ H_V_omega
    M_bm = H_omega_theta - H_V_omega.T @ H_V_inv @ H_V_theta
    M_br = H_omega_r
    M_m = H_theta - H_V_theta.T @ H_V_inv @ H_V_theta
    M_theta_r = H_theta_r
    M_r = H_r

    # v_b = dot_d_b_dt - linear velocity through conservation of linear velocity
    v_b = -1 / m_c * (H_V_omega @ dw[:3] + H_V_theta @ dw[3:6])  # omega_b = dw[:3], dot_theta = dw[3:6]

    c_V = dH_V_omega_dt @ dw[:3] + dH_V_theta_dt @ dw[3:6]  # + dH_V_r_dt @ dw[6:]
    c_b_bar = dH_V_omega_dt.T @ v_b + dH_omega_dt @ dw[:3] + dH_omega_theta_dt @ dw[3:6]  # + dH_omega_r_dt @ dw[6:]
    c_r = dH_omega_r_dt.T @ dw[:3] + dH_theta_r_dt.T @ dw[3:6]  # + dH_r_dt @ dw[6:]

    dw_new = vertcat(v_b, dw[:6], SX.zeros(3,1))
    gradient = 1 / 2 * vertcat(
        dw_new.T @ dH_dtheta1 @ dw_new,
        dw_new.T @ dH_dtheta2 @ dw_new,
        dw_new.T @ dH_dtheta3 @ dw_new
    )

    c_m_bar = dH_V_theta_dt.T @ v_b + dH_omega_theta_dt.T @ dw[:3] + dH_theta_dt @ dw[3:6] - gradient  # + dH_theta_r_dt @ dw[6:]

    c_b = c_b_bar - H_V_omega.T @ H_V_inv @ c_V
    c_m = c_m_bar - H_V_theta.T @ H_V_inv @ c_V

    M_b_tilde = M_br.T - M_r @ inv(M_br) @ M_b
    M_bm_tilde = - M_r @ inv(M_br) @ M_bm
    c_b_tilde = c_r - M_r @ inv(M_br) @ c_b

    M_tilde = vertcat(
        horzcat(M_b_tilde, M_bm_tilde),
        horzcat(M_bm.T, M_m)
    )

    C_tilde = vertcat(c_b_tilde, c_m)

    omega_s = SX([0.0, 0.0, 0.2])
    omega_rel = dw[:3] - quaternion_to_rotation_matrix(q) @ omega_s
    dq = 1/2 * Omega(omega_rel) @ q

    # Explicit dynamics: x_dot = [w_dot; M⁻¹(-C - G + u), q_dot]
    #ddw = SX.sym('q_ddot', nsd, 1)
    ddw = solve(M_tilde, tau - C_tilde)

    f_expl = vertcat(dw, ddw, dq)

    # Implicit dynamics: xdot - f_expl = 0
    f_impl = vertcat(
        dw - xdot[:6],
        M_tilde @ xdot[6:12] - C_tilde + tau,
        dq - xdot[12:]
    )
    #(xdot - f_expl)  # M * q_ddot + C - G - u

    # Define the model
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = tau
    model.name = model_name

    return model

def tau_computation(omega_b_ref, theta_ref, theta_dot_ref):
    zero = [0.0, 0.0, 0.0]
    theta = vertcat(zero, theta_ref)
    dot_theta = vertcat(zero, theta_dot_ref)
    H, dH_dt, dH_dtheta1, dH_dtheta2, dH_dtheta3, m_c = components(theta, dot_theta)

    dw = vertcat(omega_b_ref, theta_dot_ref)

    # Extract relevant sub-matrices from H and dH_dt
    H_V = H[0][0]
    H_V_omega = H[0][1]
    H_V_theta = H[0][2]
    #H_omega = H[1][1]
    #H_omega_theta = H[1][2]
    H_omega_r = H[1][3]
    #H_theta = H[2][2]
    #H_theta_r = H[2][3]
    H_r = H[3][3]

    #dH_V_dt = dH_dt[0][0]
    dH_V_omega_dt = dH_dt[0][1]
    dH_V_theta_dt = dH_dt[0][2]
    #dH_V_r_dt = dH_dt[0][3]

    dH_omega_dt = dH_dt[1][1]
    dH_omega_theta_dt = dH_dt[1][2]
    dH_omega_r_dt = dH_dt[1][3]

    dH_theta_dt = dH_dt[2][2]
    dH_theta_r_dt = dH_dt[2][3]
    #dH_r_dt = dH_dt[3][3]

    # Inverse of H_V for use later
    H_V_inv = inv(H_V)

    #M_b = H_omega - H_V_omega.T @ H_V_inv @ H_V_omega
    #M_bm = H_omega_theta - H_V_omega.T @ H_V_inv @ H_V_theta
    M_br = H_omega_r
    #M_m = H_theta - H_V_theta.T @ H_V_inv @ H_V_theta
    #M_theta_r = H_theta_r
    M_r = H_r

    # v_b = dot_d_b_dt - linear velocity through conservation of linear velocity
    v_b = -1 / m_c * (H_V_omega @ dw[:3] + H_V_theta @ dw[3:6])  # omega_b = dw[:3], dot_theta = dw[3:6]

    c_V = dH_V_omega_dt @ dw[:3] + dH_V_theta_dt @ dw[3:6]  # + dH_V_r_dt @ dw[6:]
    c_b_bar = dH_V_omega_dt.T @ v_b + dH_omega_dt @ dw[:3] + dH_omega_theta_dt @ dw[3:6]  # + dH_omega_r_dt @ dw[6:]
    c_r = dH_omega_r_dt.T @ dw[:3] + dH_theta_r_dt.T @ dw[3:6]  # + dH_r_dt @ dw[6:]

    dw_new = vertcat(v_b, dw[:6], SX.zeros(3,1))
    gradient = 1 / 2 * vertcat(
        dw_new.T @ dH_dtheta1 @ dw_new,
        dw_new.T @ dH_dtheta2 @ dw_new,
        dw_new.T @ dH_dtheta3 @ dw_new
    )

    c_m_bar = dH_V_theta_dt.T @ v_b + dH_omega_theta_dt.T @ dw[:3] + dH_theta_dt @ dw[3:6] - gradient  # + dH_theta_r_dt @ dw[6:]

    c_b = c_b_bar - H_V_omega.T @ H_V_inv @ c_V
    c_m = c_m_bar - H_V_theta.T @ H_V_inv @ c_V

    #M_b_tilde = M_br.T - M_r @ inv(M_br) @ M_b
    #M_bm_tilde = -M_r @ inv(M_br) @ M_bm
    c_b_tilde = c_r - M_r @ inv(M_br) @ c_b

    #M_tilde = vertcat(
    #    horzcat(M_b_tilde, M_bm_tilde),
    #    horzcat(M_bm.T, M_m)
    #)

    C_tilde = vertcat(c_b_tilde, c_m)

    tau = C_tilde
    #print(tau)
    return tau

def components(w, dw):

  theta, dtheta_dt = w[3:6], dw[3:6]
  # Manipulator parameters
  m1, m2, m3 = 1, 3, 2
  mi = [m1, m2, m3]
  rad1, rad2, rad3 = 0.2, 0.3, 0.4  # Radii of joints
  L1, L2, L3 = 0.2, 0.8, 0.5
  Li = [L1, L2, L3]

  # Base Satellite parameters
  m_b = 150.0  # Mass
  h_b, l_b, w_b = 1.9, 2.45, 1.41  # Height, Length, Width

  # RW parameters
  m_RW, r_RW1, r_RW2, h_RW = 5.0, 0.337/3, 0.337/2, 0.1  # Mass, small and big radius and height
  m_RWi = [m_RW, m_RW, m_RW]
  r_RW_x, r_RW_y, r_RW_z = w_b/8, l_b/8, h_b/8  # Distance to the base's CoM per axis

  theta1 = theta[0]
  theta2 = theta[1]
  theta3 = theta[2]
  dtheta1_dt = dtheta_dt[0]
  dtheta2_dt = dtheta_dt[1]
  dtheta3_dt = dtheta_dt[2]

  # Abbreviations
  theta12 = theta1 + theta2
  theta123 = theta1 + theta2 + theta3
  s1 = sin(theta1)
  c1 = cos(theta1)
  s12 = sin(theta12)
  c12 = cos(theta12)
  s123 = sin(theta123)
  c123 = cos(theta123)

  ############################### RW Jacobians ###################################
  # RW Jacobians
  J_L_RW1 = SX([
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]
  ])
  J_L_RW2 = J_L_RW1
  J_L_RW3 = J_L_RW1
  J_L_RW = [J_L_RW1, J_L_RW2, J_L_RW3]
  J_A_RW1 = SX([
      [1, 0, 0],
      [0, 0, 0],
      [0, 0, 0]
  ])
  J_A_RW2 = SX([
      [0, 0, 0],
      [0, 1, 0],
      [0, 0, 0]
  ])
  J_A_RW3 = SX([
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 1]
  ])

  J_A_RWi = [J_A_RW1, J_A_RW2, J_A_RW3]

  ############################################################################################

  # Joint 1
  J_t1 = vertcat(
      -L1 * s1,
      L1 * c1,
      0
  )
  # Time derivative
  dJ_t1_dt = vertcat(
      -dtheta1_dt * L1 * c1,
      -dtheta1_dt * L1 * s1,
      0
  )

  # Partials
  dJ_t1_dtheta1 = vertcat(
      -L1 * c1,
      -L1 * s1,
      0
  )
  dJ1_t1_dtheta2 = vertcat(
      0,
      0,
      0
  )
  dJ1_t1_dtheta3 = dJ1_t1_dtheta2

  J_r1 = vertcat(
      0,
      0,
      1
  )

  O_32 = SX.zeros((3, 2)) # Define a zero matrix for the 3x2 block
  # Construct J1 using block matrix form
  J1 = vertcat(
      horzcat(J_t1, O_32),
      horzcat(J_r1, O_32)
  )
  JL1 = J1[:3, :]
  JA1 = J1[3:, :]

  dJL1_dt = horzcat(
      dJ_t1_dt, O_32
  )

  dJL1_dtheta1 = horzcat(
      dJ_t1_dtheta1, O_32
  )
  dJL1_dtheta2 = horzcat(
      dJ1_t1_dtheta2, O_32
  )
  dJL1_dtheta3 = horzcat(
      dJ1_t1_dtheta3, O_32
  )

  # Joint 2
  J_t2_1 = vertcat(
     -L1 * s1 - L2 * s12,
     L1 * c1 + L2 * c12,
     0
  )
  # Time derivative
  dJ_t2_1_dt = vertcat(
      -dtheta1_dt * L1 * c1 - (dtheta1_dt + dtheta2_dt) * L2 * c12,
      -dtheta1_dt * L1 * s1 - (dtheta1_dt + dtheta2_dt) * L2 * s12,
      0
  )

  # Partials
  dJ_t2_1_dtheta1 = vertcat(
      -L1 * c1 - L2 * c12,
      -L1 * s1 - L2 * s12,
      0
  )
  dJ_t2_1_dtheta2 = vertcat(
      - L2 * c12,
      - L2 * s12,
      0
  )
  dJ_t2_1_dtheta3 = vertcat(
      0,
      0,
      0
  )

  J_t2_2 = vertcat(
      -L2 * s12,
      L2 * c12,
      0
  )
  # Time derivative
  dJ_t2_2_dt = vertcat(
      - (dtheta1_dt + dtheta2_dt) * L2 * c12,
      - (dtheta1_dt + dtheta2_dt) * L2 * s12,
      0
  )

  # Partials
  dJ_t2_2_dtheta1 = dJ_t2_1_dtheta2
  dJ_t2_2_dtheta2 = dJ_t2_2_dtheta1
  dJ_t2_2_dtheta3 = dJ_t2_1_dtheta3

  J_r2_1 = J_r1
  J_r2_2 = J_r1
  O_31 = SX.zeros((3, 1)) # Define a zero matrix for the 3x1 block
  # Construct J2 using block matrix form
  J2 = vertcat(
      horzcat(J_t2_1, J_t2_2, O_31),
      horzcat(J_r2_1, J_r2_2, O_31)
  )
  JL2 = J2[:3, :]
  JA2 = J2[3:, :]

  dJL2_dt = horzcat(
      dJ_t2_1_dt, dJ_t2_2_dt, O_31
  )

  dJL2_dtheta1 = horzcat(
      dJ_t2_1_dtheta1, dJ_t2_2_dtheta1, O_31
  )
  dJL2_dtheta2 = horzcat(
      dJ_t2_1_dtheta2, dJ_t2_2_dtheta2, O_31
  )
  dJL2_dtheta3 = horzcat(
      dJ_t2_1_dtheta3, dJ_t2_2_dtheta3, O_31
  )

  # Joint 3
  J_t3_1 = vertcat(
      -L1 * s1 - L2 * s12 - L3 * s123,
      L1 * c1 + L2 * c12 + L3 * c123,
      0
  )
  # Time derivative
  dJ_t3_1_dt = vertcat(
      -dtheta1_dt * L1 * c1 - (dtheta1_dt + dtheta2_dt) * L2 * c12 - (dtheta1_dt + dtheta2_dt + dtheta3_dt) * L3 * c123,
      -dtheta1_dt * L1 * s1 - (dtheta1_dt + dtheta2_dt) * L2 * s12 - (dtheta1_dt + dtheta2_dt + dtheta3_dt) * L3 * s123,
      0
  )

  # Partials
  dJ_t3_1_dtheta1 = vertcat(
      -L1 * c1 - L2 * c12 - L3 * c123,
      -L1 * s1 - L2 * s12 - L3 * s123,
      0
  )
  dJ_t3_1_dtheta2 = vertcat(
      - L2 * c12 - L3 * c123,
      - L2 * s12 - L3 * s123,
      0
  )
  dJ_t3_1_dtheta3 = vertcat(
      - L3 * c123,
      - L3 * s123,
      0
  )

  J_t3_2 = vertcat(
      -L2 * s12 - L3 * s123,
      L2 * c12 + L3 * c123,
      0
  )
  # Time derivative
  dJ_t3_2_dt = vertcat(
      - (dtheta1_dt + dtheta2_dt) * L2 * c12 - (dtheta1_dt + dtheta2_dt + dtheta3_dt) * L3 * c123,
      - (dtheta1_dt + dtheta2_dt) * L2 * s12 - (dtheta1_dt + dtheta2_dt + dtheta3_dt) * L3 * s123,
      0
  )

  # Partials
  dJ_t3_2_dtheta1 = dJ_t3_1_dtheta2
  dJ_t3_2_dtheta2 = dJ_t3_1_dtheta2
  dJ_t3_2_dtheta3 = dJ_t3_1_dtheta3

  J_t3_3 = vertcat(
      -L3 * s123,
      L3 * c123,
      0
  )
  # Time derivative
  dJ_t3_3_dt = vertcat(
      - (dtheta1_dt + dtheta2_dt + dtheta3_dt) * L3 * c123,
      - (dtheta1_dt + dtheta2_dt + dtheta3_dt) * L3 * s123,
      0
  )

  # Partials
  dJ_t3_3_dtheta1 = dJ_t3_1_dtheta3
  dJ_t3_3_dtheta2 = dJ_t3_1_dtheta3
  dJ_t3_3_dtheta3 = dJ_t3_1_dtheta3

  J_r3_1 = J_r1
  J_r3_2 = J_r1
  J_r3_3 = J_r1
  # Construct J3 using block matrix form
  J3 = vertcat(
      horzcat(J_t3_1, J_t3_2, J_t3_3),
      horzcat(J_r3_1, J_r3_2, J_r3_3)
  )
  JL3 = J3[:3, :]
  JA3 = J3[3:, :]
  dJL3_dt = horzcat(
      dJ_t3_1_dt, dJ_t3_2_dt, dJ_t3_3_dt
  )

  dJL3_dtheta1 = horzcat(
      dJ_t3_1_dtheta1, dJ_t3_2_dtheta1, dJ_t3_3_dtheta1
  )
  dJL3_dtheta2 = horzcat(
      dJ_t3_1_dtheta2, dJ_t3_2_dtheta2, dJ_t3_3_dtheta2
  )
  dJL3_dtheta3 = horzcat(
      dJ_t3_1_dtheta3, dJ_t3_2_dtheta3, dJ_t3_3_dtheta3
  )

  JLi = [JL1, JL2, JL3]
  JAi = [JA1, JA2, JA3]
  dJLi_dt = [dJL1_dt, dJL2_dt, dJL3_dt]
  dJLi_dtheta1 = [dJL1_dtheta1, dJL2_dtheta1, dJL3_dtheta1]
  dJLi_dtheta2 = [dJL1_dtheta2, dJL2_dtheta2, dJL3_dtheta2]
  dJLi_dtheta3 = [dJL1_dtheta3, dJL2_dtheta3, dJL3_dtheta3]

  ################################### r_i ######################################
  # Link 1 CoM position
  y1 = (L1 / 2) * cos(theta1)
  z1 = (L1 / 2) * sin(theta1)
  r1 = vertcat(0, y1, z1)

  # Time derivative
  dy1_dt = - dtheta1_dt * (L1 / 2) * sin(theta1)
  dz1_dt = dtheta1_dt * (L1 / 2) * cos(theta1)
  dr1_dt = vertcat(0, dy1_dt, dz1_dt)

  # Partial derivative wrt theta1
  dy1_dtheta1 = - (L1 / 2) * sin(theta1)
  dz1_dtheta1 = (L1 / 2) * cos(theta1)
  dr1_dtheta1 = vertcat(0, dy1_dtheta1, dz1_dtheta1)

  # Partial derivative wrt theta2
  dy1_dtheta2 = 0
  dz1_dtheta2 = 0
  dr1_dtheta2 = vertcat(0, dy1_dtheta2, dz1_dtheta2)

  # Partial derivative wrt theta3
  dy1_dtheta3 = 0
  dz1_dtheta3 = 0
  dr1_dtheta3 = vertcat(0, dy1_dtheta3, dz1_dtheta3)

  dr1_dtheta = horzcat(dr1_dtheta1, dr1_dtheta2, dr1_dtheta3)

  # Link 2 CoM position
  y2 = L1 * cos(theta1) + (L2 / 2) * cos(theta1 + theta2)
  z2 = L1 * sin(theta1) + (L2 / 2) * sin(theta1 + theta2)
  r2 = vertcat(0, y2, z2)

  # Tive derivative
  dy2_dt = - dtheta1_dt * L1 * sin(theta1) - \
             (dtheta1_dt + dtheta2_dt) * (L2 / 2) * sin(theta1 + theta2)
  dz2_dt = dtheta1_dt * L1 * cos(theta1) + \
           (dtheta1_dt + dtheta2_dt) * (L2 / 2) * cos(theta1 + theta2)
  dr2_dt = vertcat(0, dy2_dt, dz2_dt)

  # Partial derivative wrt theta1
  dy2_dtheta1 = -L1 * sin(theta1) - (L2 / 2) * sin(theta1 + theta2)
  dz2_dtheta1 = L1 * cos(theta1) + (L2 / 2) * cos(theta1 + theta2)
  dr2_dtheta1 = vertcat(0, dy2_dtheta1, dz2_dtheta1)

  # Partial derivative wrt theta2
  dy2_dtheta2 = - (L2 / 2) * sin(theta1 + theta2)
  dz2_dtheta2 = (L2 / 2) * cos(theta1 + theta2)
  dr2_dtheta2 = vertcat(0, dy2_dtheta2, dz2_dtheta2)

  # Partial derivative wrt theta3
  dy2_dtheta3 = 0
  dz2_dtheta3 = 0
  dr2_dtheta3 = vertcat(0, dy2_dtheta3, dz2_dtheta3)

  dr2_dtheta = horzcat(dr2_dtheta1, dr2_dtheta2, dr2_dtheta3)

  # Link 3 CoM position
  y3 = L1 * cos(theta1) + L2 * cos(theta1 + theta2) + (L3 / 2) * cos(theta1 + theta2 + theta3)
  z3 = L1 * sin(theta1) + L2 * sin(theta1 + theta2) + (L3 / 2) * sin(theta1 + theta2 + theta3)
  r3 = vertcat(0, y3, z3)

  # Time derivative
  dy3_dt = - dtheta1_dt * L1 * sin(theta1) - \
             (dtheta1_dt + dtheta2_dt) * L2 * sin(theta1 + theta2) - \
             (dtheta1_dt + dtheta2_dt + dtheta3_dt) * (L3 / 2) * sin(theta1 + theta2 + theta3)
  dz3_dt = dtheta1_dt * L1 * cos(theta1) + \
           (dtheta1_dt + dtheta2_dt) * L2 * cos(theta1 + theta2) + \
           (dtheta1_dt + dtheta2_dt + dtheta3_dt) * (L3 / 2) * cos(theta1 + theta2 + theta3)
  dr3_dt = vertcat(0, dy3_dt, dz3_dt)


  # Partial derivative wrt theta1
  dy3_dtheta1 = -L1 * sin(theta1) - L2 * sin(theta1 + theta2) - (L3 / 2) * sin(theta1 + theta2 + theta3)
  dz3_dtheta1 = L1 * cos(theta1) + L2 * cos(theta1 + theta2) + (L3 / 2) * cos(theta1 + theta2 + theta3)
  dr3_dtheta1 = vertcat(0, dy3_dtheta1, dz3_dtheta1)

  # Partial detivative wrt theta2
  dy3_dtheta2 = - L2 * sin(theta1 + theta2) - (L3 / 2) * sin(theta1 + theta2 + theta3)
  dz3_dtheta2 = L2 * cos(theta1 + theta2) + (L3 / 2) * cos(theta1 + theta2 + theta3)
  dr3_dtheta2 = vertcat(0, dy3_dtheta2, dz3_dtheta2)

  # Partial derivative wrt theta3
  dy3_dtheta3 = - (L3 / 2) * sin(theta1 + theta2 + theta3)
  dz3_dtheta3 = (L3 / 2) * cos(theta1 + theta2 + theta3)
  dr3_dtheta3 = vertcat(0, dy3_dtheta3, dz3_dtheta3)

  dr3_dtheta = horzcat(dr3_dtheta1, dr3_dtheta2, dr3_dtheta3)

  ri = [r1, r2, r3]
  dri_dt = [dr1_dt, dr2_dt, dr3_dt]
  dri_dtheta1 = [dr1_dtheta1, dr2_dtheta1, dr3_dtheta1]
  dri_dtheta2 = [dr1_dtheta2, dr2_dtheta2, dr3_dtheta2]
  dri_dtheta3 = [dr1_dtheta3, dr2_dtheta3, dr3_dtheta3]

  ############################# Inertia Matrices ###############################

  # Base
  I_b = SX([[120, 30, -40],
            [30, 70, 20],
            [-40, 20, 100]])

  # RWs
  # I_RW1_base, I_RW2_base, I_RW3_base RW inertia at base frame Sigma_B
  I_RW1_base, I_RW2_base, I_RW3_base = \
              RW_inertia_matrix(m_RW, r_RW1, r_RW2, h_RW, r_RW_x, r_RW_y, r_RW_z)
  I_RWi = [I_RW1_base, I_RW2_base, I_RW3_base]

  # Manipulator
  I1 = rotate_inertia_tensor(joint_inertia_tensor(m1, rad1, L1), axis='y', theta_deg=90)
  I2 = rotate_inertia_tensor(joint_inertia_tensor(m2, rad2, L2), axis='y', theta_deg=90)
  I3 = rotate_inertia_tensor(joint_inertia_tensor(m3, rad3, L3), axis='y', theta_deg=90)
  I1_base, dI1_base_dt, dI1_base_dtheta = parallel_axis_theorem(I1, m1, r1, dr1_dt, dr1_dtheta)
  I2_base, dI2_base_dt, dI2_base_dtheta = parallel_axis_theorem(I2, m2, r2, dr2_dt, dr2_dtheta)
  I3_base, dI3_base_dt, dI3_base_dtheta = parallel_axis_theorem(I3, m3, r3, dr3_dt, dr3_dtheta)

  Ii = [I1_base, I2_base, I3_base]
  dIi_dt = [dI1_base_dt, dI2_base_dt, dI3_base_dt]
  dIi_dtheta1 = [dI1_base_dtheta[0], dI2_base_dtheta[0], dI3_base_dtheta[0]]
  dIi_dtheta2 = [dI1_base_dtheta[1], dI2_base_dtheta[1], dI3_base_dtheta[1]]
  dIi_dtheta3 = [dI1_base_dtheta[2], dI2_base_dtheta[2], dI3_base_dtheta[2]]

  ################################ Centroid ####################################
  # Mass (m_c)
  m_c = sum(mi) + m_b + 3 * m_RW

  # Inertia (I_c)
  I_RW_3x3 = np.eye(3)  # 3x3 inertia matrix for the reaction wheels (identity matrix as placeholder)
  I_c = sum(Ii) + I_b + sum(I_RWi)
  dI_c_dt = sum(dIi_dt)
  dI_c_dtheta1 = sum(dIi_dtheta1)
  dI_c_dtheta2 = sum(dIi_dtheta2)
  dI_c_dtheta3 = sum(dIi_dtheta3)

  # CoM position (r_c)
  r_RWi = [
      vertcat(r_RW_x, 0, 0),  # Reaction wheel 1 position (x-axis)
      vertcat(0, r_RW_y, 0),  # Reaction wheel 2 position (y-axis)
      vertcat(0, 0, r_RW_z)   # Reaction wheel 3 position (z-axis)
  ]
  r_c = (sum(mi[i] * ri[i] for i in range(len(mi))) +
         sum(m_RWi[i] * r_RWi[i] for i in range(len(m_RWi)))) / m_c
  dr_c_dt = (sum(mi[i] * dri_dt[i] for i in range(len(mi)))) / m_c

  dr_c_dtheta1 = (sum(mi[i] * dri_dtheta1[i] for i in range(len(mi)))) / m_c
  dr_c_dtheta2 = (sum(mi[i] * dri_dtheta2[i] for i in range(len(mi)))) / m_c
  dr_c_dtheta3 = (sum(mi[i] * dri_dtheta3[i] for i in range(len(mi)))) / m_c

  # Total moment of inertia (J_c)
  J_c = (sum(mi[i] * JLi[i] for i in range(len(mi)))) / m_c
  dJ_c_dt = (sum(mi[i] * dJLi_dt[i] for i in range(len(mi)))) / m_c

  dJ_c_dtheta1 = (sum(mi[i] * dJLi_dtheta1[i] for i in range(len(mi)))) / m_c
  dJ_c_dtheta2 = (sum(mi[i] * dJLi_dtheta2[i] for i in range(len(mi)))) / m_c
  dJ_c_dtheta3 = (sum(mi[i] * dJLi_dtheta3[i] for i in range(len(mi)))) / m_c

  ################################ H matrices ##################################
  H_V = m_c*SX.eye(3)
  H_V_omega = -m_c*skew_symmetric(r_c)
  H_V_theta = m_c*J_c
  H_V_r = SX.zeros(3, 3)
  H_omega = I_c + sum(skew_symmetric(ri[i]).T*skew_symmetric(ri[i])*mi[i] for i in range(len(mi)))

  H_omega_theta = sum(Ii[i]*JAi[i] + mi[i]*skew_symmetric(ri[i])*JLi[i] for i in range(len(mi)))

  H_omega_r = sum(I_RWi[i]*J_A_RWi[i] for i in range(len(I_RWi)))

  H_theta = sum(JLi[i].T*mi[i]*JLi[i] + JAi[i].T*Ii[i]*JAi[i] for i in range(len(mi)))

  H_theta_r = SX.zeros(3, 3)

  H_r = sum(J_A_RWi[i].T*I_RWi[i] for i in range(len(m_RWi)))

  H = [
      [H_V,         H_V_omega,       H_V_theta,     H_V_r],
      [H_V_omega.T, H_omega,         H_omega_theta, H_omega_r],
      [H_V_theta.T, H_omega_theta.T, H_theta,       H_theta_r],
      [H_V_r.T,     H_omega_r.T,     H_theta_r.T,   H_r]
  ]

  ############################## H dot matrices ################################
  dH_V_dt = SX.zeros(3,3)
  dH_V_omega_dt = -m_c*skew_symmetric(dr_c_dt)
  dH_V_theta_dt = m_c*dJ_c_dt
  dH_omega_dt = dI_c_dt + sum(skew_symmetric(dri_dt[i]).T @ skew_symmetric(ri[i])*mi[i] + \
                              skew_symmetric(ri[i]).T @ skew_symmetric(dri_dt[i])*mi[i] for i in range(len(mi)))

  dH_omega_theta_dt = sum(dIi_dt[i]*JAi[i] + \
                          mi[i]*skew_symmetric(dri_dt[i]) @ JLi[i] + \
                          mi[i]*skew_symmetric(ri[i]) @ dJLi_dt[i] for i in range(len(mi)))

  dH_omega_r_dt = SX.zeros(3,3)
  dH_theta_dt = sum(JAi[i].T @ dIi_dt[i] @ JAi[i] + \
                    dJLi_dt[i].T * mi[i] @ JLi[i] + \
                    JLi[i].T * mi[i] @ dJLi_dt[i] for i in range(len(mi)))

  dH_r_dt = SX.zeros(3,3)
  dH_theta_r_dt = SX.zeros(3,3)
  dH_V_r_dt = SX.zeros(3,3)

  dH_dt = [
      [dH_V_dt,         dH_V_omega_dt,       dH_V_theta_dt,     dH_V_r_dt],
      [dH_V_omega_dt.T, dH_omega_dt,         dH_omega_theta_dt, dH_omega_r_dt],
      [dH_V_theta_dt.T, dH_omega_theta_dt.T, dH_theta_dt,       dH_theta_r_dt],
      [dH_V_r_dt.T,     dH_omega_r_dt.T,     dH_theta_r_dt.T,   dH_r_dt]
  ]

  ################################ H partials ##################################
  dH_V_dtheta1 = SX.zeros(3,3)
  dH_V_omega_dtheta1 = -m_c*skew_symmetric(dr_c_dtheta1)
  dH_V_theta_dtheta1 = m_c*dJ_c_dtheta1
  dH_omega_dtheta1 = dI_c_dtheta1 + sum(skew_symmetric(dri_dtheta1[i]).T @ skew_symmetric(ri[i])*mi[i] + \
                                        skew_symmetric(ri[i]).T @ skew_symmetric(dri_dtheta1[i])*mi[i] for i in range(len(mi)))
  dH_omega_theta_dtheta1 = sum(dIi_dtheta1[i] @ JAi[i] +
                               mi[i]*skew_symmetric(dri_dtheta1[i]) @ JLi[i] +
                               mi[i]*skew_symmetric(ri[i]) @ dJLi_dtheta1[i] for i in range(len(mi)))
  dH_omega_r_dtheta1 = SX.zeros(3,3)
  dH_theta_dtheta1 = sum(JAi[i].T @ dIi_dtheta1[i] @ JAi[i] +
                         dJLi_dtheta1[i].T * mi[i] @ JLi[i] +
                         JLi[i].T * mi[i] @ dJLi_dtheta1[i] for i in range(len(mi)))
  dH_r_dtheta1 = SX.zeros(3,3)
  dH_theta_r_dtheta1 = SX.zeros(3,3)
  dH_V_r_dtheta1 = SX.zeros(3,3)

  dH_dtheta1 = vertcat(
      horzcat(dH_V_dtheta1,         dH_V_omega_dtheta1,       dH_V_theta_dtheta1,     dH_V_r_dtheta1),
      horzcat(dH_V_omega_dtheta1.T, dH_omega_dtheta1,         dH_omega_theta_dtheta1, dH_omega_r_dtheta1),
      horzcat(dH_V_theta_dtheta1.T, dH_omega_theta_dtheta1.T, dH_theta_dtheta1,       dH_theta_r_dtheta1),
      horzcat(dH_V_r_dtheta1.T,     dH_omega_r_dtheta1.T,     dH_theta_r_dtheta1.T,   dH_r_dtheta1)
  )

  dH_V_dtheta2 = SX.zeros((3,3))
  dH_V_omega_dtheta2 = -m_c*skew_symmetric(dr_c_dtheta2)
  dH_V_theta_dtheta2 = m_c*dJ_c_dtheta2
  dH_omega_dtheta2 = dI_c_dtheta2 + sum(skew_symmetric(dri_dtheta2[i]).T @ skew_symmetric(ri[i])*mi[i] + \
                                        skew_symmetric(ri[i]).T @ skew_symmetric(dri_dtheta2[i])*mi[i] for i in range(len(mi)))
  dH_omega_theta_dtheta2 = sum(dIi_dtheta2[i] @ JAi[i] + \
                               mi[i]*skew_symmetric(dri_dtheta2[i]) @ JLi[i] + \
                               mi[i]*skew_symmetric(ri[i]) @ dJLi_dtheta2[i] for i in range(len(mi)))
  dH_omega_r_dtheta2 = SX.zeros((3,3))
  dH_theta_dtheta2 = sum(JAi[i].T @ dIi_dtheta2[i] @ JAi[i] + \
                         dJLi_dtheta2[i].T * mi[i] @ JLi[i] + \
                         JLi[i].T * mi[i] @ dJLi_dtheta2[i] for i in range(len(mi)))
  dH_r_dtheta2 = SX.zeros((3,3))
  dH_theta_r_dtheta2 = SX.zeros((3,3))
  dH_V_r_dtheta2 = SX.zeros((3,3))

  dH_dtheta2 = vertcat(
      horzcat(dH_V_dtheta2,         dH_V_omega_dtheta2,       dH_V_theta_dtheta2,     dH_V_r_dtheta2),
      horzcat(dH_V_omega_dtheta2.T, dH_omega_dtheta2,         dH_omega_theta_dtheta2, dH_omega_r_dtheta2),
      horzcat(dH_V_theta_dtheta2.T, dH_omega_theta_dtheta2.T, dH_theta_dtheta2,       dH_theta_r_dtheta2),
      horzcat(dH_V_r_dtheta2.T,     dH_omega_r_dtheta2.T,     dH_theta_r_dtheta2.T,   dH_r_dtheta2)
  )

  dH_V_dtheta3 = SX.zeros((3,3))
  dH_V_omega_dtheta3 = -m_c*skew_symmetric(dr_c_dtheta3)
  dH_V_theta_dtheta3 = m_c*dJ_c_dtheta3
  dH_omega_dtheta3 = dI_c_dtheta3 + sum(skew_symmetric(dri_dtheta3[i]).T @ skew_symmetric(ri[i])*mi[i] + \
                                        skew_symmetric(ri[i]).T @ skew_symmetric(dri_dtheta3[i])*mi[i] for i in range(len(mi)))
  dH_omega_theta_dtheta3 = sum(dIi_dtheta3[i] @ JAi[i] + \
                               mi[i]*skew_symmetric(dri_dtheta3[i]) @ JLi[i] + \
                               mi[i]*skew_symmetric(ri[i]) @ dJLi_dtheta3[i] for i in range(len(mi)))
  dH_omega_r_dtheta3 = SX.zeros((3,3))
  dH_theta_dtheta3 = sum(JAi[i].T @ dIi_dtheta3[i] @ JAi[i] + \
                         dJLi_dtheta3[i].T * mi[i] @ JLi[i] + \
                         JLi[i].T * mi[i] @ dJLi_dtheta3[i] for i in range(len(mi)))
  dH_r_dtheta3 = SX.zeros((3,3))
  dH_theta_r_dtheta3 = SX.zeros((3,3))
  dH_V_r_dtheta3 = SX.zeros((3,3))

  dH_dtheta3 = vertcat(
      horzcat(dH_V_dtheta3,         dH_V_omega_dtheta3,       dH_V_theta_dtheta3,     dH_V_r_dtheta3),
      horzcat(dH_V_omega_dtheta3.T, dH_omega_dtheta3,         dH_omega_theta_dtheta3, dH_omega_r_dtheta3),
      horzcat(dH_V_theta_dtheta3.T, dH_omega_theta_dtheta3.T, dH_theta_dtheta3,       dH_theta_r_dtheta3),
      horzcat(dH_V_r_dtheta3.T,     dH_omega_r_dtheta3.T,     dH_theta_r_dtheta3.T,   dH_r_dtheta3)
  )

  return H, dH_dt, dH_dtheta1, dH_dtheta2, dH_dtheta3, m_c