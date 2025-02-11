from acados_template import AcadosSim, AcadosSimSolver
from export_SA_model import export_SA_model

# create ocp object to formulate the simulation problem
sim = AcadosSim()

def export_SA_integrator():
    # Simulation options
    Ts = 0.01  # Sampling time (seconds)

    model = export_SA_model()

    # Set model
    sim.model = model

    # Dimensions
    nx = model.x.rows()  # Number of states
    nu = model.u.rows()  # Number of control inputs

    if nx <= 0 or nu <= 0:
        raise ValueError("Model dimensions are invalid. Check your dynamics model.")


    # Solver options
    sim.solver_options.integrator_type = 'IRK'  # Implicit Runge-Kutta method

    sim.solver_options.num_stages = 4  # Number of stages in the IRK method
    sim.solver_options.num_steps = 2   # Number of integration steps per sampling time

    # Set simulation horizon
    sim.solver_options.T = Ts

    # Create the AcadosSimSolver
    acados_integrator = AcadosSimSolver(sim, json_file='acados_sim_' + model.name + '.json')

    return acados_integrator
