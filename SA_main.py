import numpy as np
import matplotlib.pyplot as plt
from acados_template import AcadosOcp, AcadosOcpSolver
from acados_template import AcadosModel
from casadi import SX, DM, vertcat, horzcat, norm_fro, eig_symbolic
from export_SA_solver import export_SA_solver
from export_SA_integrator import export_SA_integrator
from export_SA_model import export_SA_model, tau_computation

def skew_symmetric(v):
    """Assume a vector v = [v1, v2, v3]"""
    return vertcat(
        horzcat(0, -v[2], v[1]),
        horzcat(v[2], 0, -v[0]),
        horzcat(-v[1], v[0], 0)
    )

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

def run_SA_control(N_sim):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # Export your model
    model = export_SA_model()
    ocp.model = model

    # Set dimensions
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu

    ocp_solver = export_SA_solver()
    integrator = export_SA_integrator()

    # Simulation parameters
    xcurrent = np.array([0.0, 0.0, 0.0,
                         0.05, 0.4, 0.05,
                         0.1, 0.0, 0.2,
                         0.0, 0.0, 0.0,
                         0.1, 0.1, 0.1, 1.0])  # Initial state: [r_b, theta, omega_b, theta_dot, q]

    #simX = np.zeros((N_sim + 1, nx))
    simX = np.zeros((N_sim + 1, nx))
    simU = np.zeros((N_sim, nu))
    omega_matrix_norm = np.zeros(N_sim)
    Omega_eigvals = np.zeros((N_sim, 4), dtype = "complex128")

    timings = np.zeros((N_sim,))
    simX[0, :] = xcurrent

    #state_noise_mean = 0.0  # Mean of the noise
    #state_noise_std = 0.01  # Standard deviation of the noise

    state_noise_mean = np.zeros_like(xcurrent)  # Mean noise for all states (zero-centered)
    state_noise_std = np.array([
        0.001, 0.0001, 0.01,            # Noise for r_b
        0.001, 0.001, 0.001,            # Noise for theta
        0.0001, 0.0001, 0.0001,         # Noise for omega_b
        0.001, 0.00001, 0.00001,        # Noise for theta_dot
        0.001, 0.001, 0.001, 0.001      # Noise for q
    ])

    for i in range(N_sim):
        # Solve OCP to get the optimal control for the current state
        #simU[i, :] = ocp_solver.solve_for_x0(x0_bar=xcurrent)

        #if i == N_sim//2:
        #    desired_x = np.array([0.0, 0.0, 0.2,
        #                          0.3, 0.4, 0.5,
        #                          0.0, 0.0, 0.2,
        #                          0.0, 0.0, 0.0,
        #                          0.0, 0.0, 0.0, 1.0])  # Target state
        #    theta_ref = [0.3, 0.4, 0.5]
        #    theta_dot_ref = [0.0, 0.0, 0.0]
        #    omega_b_ref = [0.0, 0.0, 0.2]
        #    tau_ref = tau_computation(omega_b_ref, theta_ref, theta_dot_ref)
        #    tau_ref = np.reshape(DM(tau_ref), (1, -1))

        #    yref = np.hstack((desired_x, tau_ref[0]))  # State + control reference
        #    for stage in range(ocp_solver.acados_ocp.dims.N):
        #        ocp_solver.set(stage, "yref", yref)
        #    ocp_solver.set(ocp_solver.acados_ocp.dims.N, "yref", desired_x)

        ##################################################################################
        # Set initial state for the solver
        ocp_solver.set(0, "lbx", xcurrent)
        ocp_solver.set(0, "ubx", xcurrent)

        # Solve the OCP
        status = ocp_solver.solve()
        if status != 0:
            raise RuntimeError(f"OCP Solver failed at step {i} with status {status}")

        # Get the first control input
        simU[i, :] = ocp_solver.get(0, "u")

        #u_phys = ocp_solver.get(0, "u")  # Normalized control value from the solver
        #u_max_abs = max(abs(u) for u in u_phys)
        #simU[i, :] = [u / u_max_abs for u in u_phys]
        ##################################################################################

        timings[i] = ocp_solver.get_stats("time_tot")

        print(f"Step {i}: Control = {simU[i, :]}")

        # Simulate system dynamics using the integrator
        xcurrent = integrator.simulate(x=xcurrent, u=simU[i, :])

        # Add noise to the state
        noise = np.random.normal(state_noise_mean, state_noise_std)
        xcurrent += noise

        # compute ||Omega(omega_rel)||_2 and store eigenvalues of the Omega matrix
        omega_s = SX([0.0, 0.0, 0.2])
        omega_rel = relative_angular_velocity(xcurrent[6:9], omega_s, xcurrent[12:])
        Omega_matrix = Omega(omega_rel)
        omega_matrix_norm[i] = norm_fro(Omega_matrix)
        Omega_eigvals[i, :] = np.linalg.eigvals(DM(Omega_matrix))

        simX[i + 1, :] = xcurrent
        print("##########################################################################################")
        print(f"Step {i}: State = {simX[i + 1, :]}")
        print("##########################################################################################")

    return simX, simU, timings, omega_matrix_norm, Omega_eigvals

if __name__ == '__main__':
    # Run the simulation
    tf = .350
    N_horizon = 8
    dt = tf/N_horizon  # Time step
    #dt = 0.01
    total_time = 100  # Total simulation time
    N_sim = int(total_time / dt)
    time = np.linspace(0, total_time, N_sim)
    state_traj, control_traj, solve_times, omega_matrix_norm, Omega_eigvals = run_SA_control(N_sim)

    print("#############################################################")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))

    # Plot 1: r_b (first 3 components)
    plt.subplot(5, 1, 1)
    plt.plot(time, state_traj[:-1, 0], label="r_b_x", color='b')
    plt.plot(time, state_traj[:-1, 1], label="r_b_y", color='g')
    plt.plot(time, state_traj[:-1, 2], label="r_b_z", color='r')
    plt.legend()
    #plt.title("Position of the base")
    plt.xlabel("Time Step")
    plt.ylabel("Base Position")
    plt.grid()

    # Plot 2: theta (next 3 components)
    plt.subplot(5, 1, 2)
    plt.plot(time, state_traj[:-1, 3], label="theta_x", color='b')
    plt.plot(time, state_traj[:-1, 4], label="theta_y", color='g')
    plt.plot(time, state_traj[:-1, 5], label="theta_z", color='r')
    plt.legend()
    #plt.title("Joint angles")
    plt.xlabel("Time Step")
    plt.ylabel("Joint angles")
    plt.grid()

    # Plot 3: omega_b (next 3 components)
    plt.subplot(5, 1, 3)
    plt.plot(time, state_traj[:-1, 6], label="omega_b_x", color='b')
    plt.plot(time, state_traj[:-1, 7], label="omega_b_y", color='g')
    plt.plot(time, state_traj[:-1, 8], label="omega_b_z", color='r')
    plt.legend()
    #plt.title("Angular Velocity of the base")
    plt.xlabel("Time Step")
    plt.ylabel("Base Ang. Vel.")
    plt.grid()

    # Plot 4: theta_dot (next 3 components)
    plt.subplot(5, 1, 4)
    plt.plot(time, state_traj[:-1, 9], label="theta_dot_x", color='b')
    plt.plot(time, state_traj[:-1, 10], label="theta_dot_y", color='g')
    plt.plot(time, state_traj[:-1, 11], label="theta_dot_z", color='r')
    plt.legend()
    #plt.title("Angular Velocity of joint angle")
    plt.xlabel("Time Step")
    plt.ylabel("Joint Ang. Vel.")
    plt.grid()

    # Plot 5: Quaternion (last 4 components)
    plt.subplot(5, 1, 5)
    plt.plot(time, state_traj[:-1, 12], label="q_x", color='k')  # Scalar part
    plt.plot(time, state_traj[:-1, 13], label="q_y", color='b')  # x-component
    plt.plot(time, state_traj[:-1, 14], label="q_z", color='g')  # y-component
    plt.plot(time, state_traj[:-1, 15], label="q_o", color='r')  # z-component
    plt.legend()
    #plt.title("Quaternion")
    plt.xlabel("Time Step")
    plt.ylabel("Quaternion")
    plt.grid()

    # Plot the values of the 2-norm of Omega(omega_rel)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, omega_matrix_norm, label="||Omega(omega_rel)||_2", color='k')  # Scalar part
    plt.legend()
    plt.title("||Omega(omega_rel)||_2")
    plt.xlabel("Time Step")
    plt.ylabel("||.||_2 Values")
    plt.grid()

    # Plot the eigenvalues in the complex plane
    real_parts = Omega_eigvals.real
    imag_parts = Omega_eigvals.imag
    plt.subplot(2, 1, 2)
    plt.scatter(real_parts, imag_parts, color='blue', label="Eigenvalues")
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.title("Eigenvalues of Omega(omega_rel) in the Complex Plane")
    plt.grid(True)
    plt.legend()

    # Plot solver timings
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 1, 1)
    plt.plot(time, solve_times, label="Solver Time", color='r')  # Total time per iteration
    plt.xlabel("Time Step")
    plt.ylabel("Solver Time (s)")
    plt.title("Solver Time Per Iteration")
    plt.grid(True)
    plt.legend()

    # Plot control trajectory (2 separate plots)
    # Plot 1: tau_r (first 3 components of control)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, control_traj[:, 0], label="tau_r_x", color='b')
    plt.plot(time, control_traj[:, 1], label="tau_r_y", color='g')
    plt.plot(time, control_traj[:, 2], label="tau_r_z", color='r')
    plt.legend()
    plt.title("RW Torque (tau_r)")
    plt.xlabel("Time Step")
    plt.ylabel("Control Value (tau_r)")
    plt.grid()

    # Plot 2: tau_m (last 3 components of control)
    plt.subplot(2, 1, 2)
    plt.plot(time, control_traj[:, 3], label="tau_m_x", color='b')
    plt.plot(time, control_traj[:, 4], label="tau_m_y", color='g')
    plt.plot(time, control_traj[:, 5], label="tau_m_z", color='r')
    plt.legend()
    plt.title("Manipulator Torque (tau_m)")
    plt.xlabel("Time Step")
    plt.ylabel("Control Value (tau_m)")
    plt.grid()

    plt.tight_layout()
    plt.show()
