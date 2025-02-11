import numpy as np
import matplotlib.pyplot as plt
from acados_template import AcadosOcp, AcadosOcpSolver
from acados_template import AcadosModel
from casadi import SX, DM, vertcat, horzcat, norm_fro
from phase_A_solver import phase_A_solver
from phase_A_integrator import phase_A_integrator
from phase_A_model import phase_A_model, tau_computation

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

def phase_A_control(N_sim):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # Export your model
    model = phase_A_model()
    ocp.model = model

    # Set dimensions
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu

    ocp_solver = phase_A_solver()
    integrator = phase_A_integrator()

    # Simulation parameters
    xcurrent = np.array([0.0, 0.0, 0.0,
                         0.1, 0.05, 0.01,
                         0.5, 0.2, 0.4, -0.8])  # Initial state: [r_b, omega_b, q]

    #simX = np.zeros((N_sim + 1, nx))
    simX = np.zeros((N_sim + 1, nx))
    simU = np.zeros((N_sim, nu))
    omega_matrix_norm = np.zeros(N_sim)
    Omega_eigvals = np.zeros((N_sim, 4), dtype = "complex128")

    timings = np.zeros((N_sim,))
    simX[0, :] = xcurrent

    state_noise_mean = np.zeros_like(xcurrent)  # Mean noise for all states (zero-centered)
    state_noise_std = np.array([
        0.001, 0.0001, 0.01,        # Noise for r_b
        0.0001, 0.0001, 0.0001,     # Noise for omega_b
        0.001, 0.001, 0.001, 0.001  # Noise for q
    ])

    for i in range(N_sim):
        # Solve OCP to get the optimal control for the current state
        #simU[i, :] = ocp_solver.solve_for_x0(x0_bar=xcurrent)

        #if i == N_sim//2:
        #    desired_x = np.array([0.0, 0.0, 10,
        #                          0.0, 0.0, 0.25,
        #                          0.0, 0.0, 0.0, 1.0])  # Target state
        #    theta_ref = [0.05, 0.4, 0.05]
        #    theta_dot_ref = [0.0, 0.0, 0.0]
        #    omega_b_ref = [0.0, 0.0, 0.25]
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
        omega_rel = relative_angular_velocity(xcurrent[3:6], omega_s, xcurrent[6:])
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
    N_horizon = 50
    dt = tf / N_horizon  # Time step
    # dt = 0.01
    total_time = 80  # Total simulation time
    N_sim = int(total_time / dt)
    time = np.linspace(0, total_time, N_sim)
    state_traj, control_traj, solve_times, omega_matrix_norm, Omega_eigvals = phase_A_control(N_sim)

    # Extract quaternion components from state_traj
    q_x = state_traj[:, 6]  # x-component
    q_y = state_traj[:, 7]  # y-component
    q_z = state_traj[:, 8]  # z-component
    q_o = state_traj[:, 9]  # Scalar part

    # Stack them together to form a 2D array, each row is a quaternion
    q_seq = np.stack((q_o, q_x, q_y, q_z), axis=1)

    # Save to CSV
    np.savetxt("q_seq.csv", q_seq, delimiter=",")

    print("#############################################################")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))

    # Plot 1: omega_b (next 3 components)
    plt.subplot(2, 1, 1)
    plt.plot(time, state_traj[:-1, 3], label="omega_b_x", color='b')
    plt.plot(time, state_traj[:-1, 4], label="omega_b_y", color='g')
    plt.plot(time, state_traj[:-1, 5], label="omega_b_z", color='r')
    plt.legend()
    plt.title("Angular Velocity of the base (omega_b)")
    plt.xlabel("Time Step")
    plt.ylabel("Angular Velocity Value")
    plt.grid()

    # Plot 2: Quaternion (last 4 components)
    plt.subplot(2, 1, 2)
    plt.plot(time, state_traj[:-1, 6], label="q_x", color='k')  # Scalar part
    plt.plot(time, state_traj[:-1, 7], label="q_y", color='b')  # x-component
    plt.plot(time, state_traj[:-1, 8], label="q_z", color='g')  # y-component
    plt.plot(time, state_traj[:-1, 9], label="q_o", color='r')  # z-component
    plt.legend()
    plt.title("Quaternion (q)")
    plt.xlabel("Time Step")
    plt.ylabel("Quaternion Component Value")
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

    # Plot control trajectory (2 separate plots)
    # Plot 1: tau_r (first 3 components of control)
    plt.figure(figsize=(10, 6))
    plt.plot(time, control_traj[:, 0], label="tau_r_x", color='b')
    plt.plot(time, control_traj[:, 1], label="tau_r_y", color='g')
    plt.plot(time, control_traj[:, 2], label="tau_r_z", color='r')
    plt.legend()
    plt.title("RW Torque (tau_r)")
    plt.xlabel("Time Step")
    plt.ylabel("Control Value (tau_r)")
    plt.grid()

    plt.tight_layout()

    plt.show()
