
import numpy as np
from scipy.optimize import minimize


# --------------------------------------------------
# 1. READ SPIN CONFIGURATIONS
# --------------------------------------------------

def read_spin_config(filename, L):

    N = L * L

    with open(filename, "r") as file:
        data = file.read()

    values = np.array(
        data.replace("[", " ").replace("]", " ").split(),
        dtype=int
    )

    if len(values) % N != 0:
        raise ValueError("Incorrect number of spins in file.")

    M = len(values) // N

    spin = values.reshape(M, N)

    # Convert each flattened configuration back to L x L.
    spin = spin.reshape(M, L, L)

    return spin


# --------------------------------------------------
# 2. CALCULATE NEAREST-NEIGHBOUR SPINS
# --------------------------------------------------

def neighbour_sum(spin):

    # Periodic boundary conditions.
    north = np.roll(spin, 1, axis=1)
    south = np.roll(spin, -1, axis=1)

    west = np.roll(spin, 1, axis=2)
    east = np.roll(spin, -1, axis=2)

    return north + east + south + west


# --------------------------------------------------
# 3. PSEUDOLIKELIHOOD COST FUNCTION
# --------------------------------------------------

def cost_func(theta, spin):

    # Two parameters:
    J, h = theta

    M, L, _ = spin.shape
    N = L * L

    # Sum of nearest neighbours.
    neighbours = neighbour_sum(spin)

    # Effective local field.
    local_field = J * neighbours + h

    # Negative log pseudolikelihood.
    z = 2 * spin * local_field

    loss = np.sum(np.logaddexp(0, -z)) / (M * N)

    # ----------------------------------------------
    # Analytical gradients
    # ----------------------------------------------

    d = -2 * spin / (1 + np.exp(np.clip(z, -500, 500)))

    grad_J = np.sum(d * neighbours) / (M * N)

    grad_h = np.sum(d) / (M * N)

    gradient = np.array([grad_J, grad_h])

    return loss, gradient


# --------------------------------------------------
# 4. MAIN
# --------------------------------------------------

def main():

    L = 8

    # Read spin configurations.
    spin = read_spin_config("C:/Users/oscar/Desktop/SchoolRelated/ProjectCourseHT26/IsingModelProject/spin_config.txt", L)

    M = len(spin)

    print("Number of configurations:", M)
    print("Lattice size:", L, "x", L)

    # Initial parameter guess.
    theta0 = np.array([0.0, 0.0])

    # Optimize.
    results = minimize(
        cost_func,
        theta0,
        args=(spin,),
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": 2000,
            "ftol": 1e-12
        }
    )

    # Extract estimated parameters.
    J_est, h_est = results.x

    print("\nOptimization successful:", results.success)
    print("Optimization message:", results.message)

    print("\nEstimated coupling J:", J_est)
    print("Estimated magnetic field h:", h_est)

    print("\nFinal cost:", results.fun)

    return J_est, h_est, results


if __name__ == "__main__":

    J_est, h_est, results = main()
