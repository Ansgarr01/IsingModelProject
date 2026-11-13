
import numpy as np
import re
from scipy.optimize import minimize
from scipy.special import expit
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. EMPIRICAL OBSERVABLES
# --------------------------------------------------

def avg_mag(spin):
    """
    Average magnetization of every spin.

    spin: (M, N)
    Returns: (N,)
    """
    return np.mean(spin, axis=0)


def avg_C(spin):
    """
    Average pairwise spin correlations.

    spin: (M, N)
    Returns: (N, N)
    """
    M, N = spin.shape

    C_tot = np.zeros((N, N))

    for mu in range(M):
        sspin = spin[mu]

        for i in range(N):
            for j in range(i + 1, N):
                C_tot[i, j] += sspin[i] * sspin[j]
                C_tot[j, i] += sspin[i] * sspin[j]

    return C_tot / M


# --------------------------------------------------
# 2. READ SPIN CONFIGURATIONS
# --------------------------------------------------

def read_spin_config(filename, L):

    N = L * L

    with open(filename, "r") as file:
        data = file.read()

    # Extract all +1 and -1 spin values.
    values = re.findall(r'-1|(?<!\d)1(?!\d)', data)

    values = np.array(values, dtype=int)

    if len(values) % N != 0:
        raise ValueError(
            "Number of spins in file is not divisible by L*L. "
            "Check the file format and lattice size."
        )

    M = len(values) // N

    # Each row represents one spin configuration.
    spin = values.reshape(M, N)

    return spin


# --------------------------------------------------
# 3. CONVERT PARAMETERS INTO J AND h
# --------------------------------------------------

def unpack_parameters(theta, N):

    n_J = N * (N - 1) // 2

    h = theta[:N]

    J = np.zeros((N, N))

    # Extract upper triangular coupling parameters.
    indices = np.triu_indices(N, k=1)

    J[indices] = theta[N:N + n_J]

    # Coupling matrix must be symmetric.
    J += J.T

    return J, h


# --------------------------------------------------
# 4. PSEUDOLIKELIHOOD COST FUNCTION
# --------------------------------------------------

def cost_func(theta, spin, L, lambda_J=1e-4, lambda_h=1e-2):
    """
    Pseudolikelihood objective with:
      - L2 regularization on J
      - H1 (first-order Tikhonov) regularization on h

    spin has shape (number_of_samples, L*L).
    """

    M, N = spin.shape

    J, h = unpack_parameters(theta, N)

    # Pseudolikelihood
    local_field = spin @ J + h
    z = 2 * spin * local_field

    loss = np.sum(np.logaddexp(0, -z)) / (M * N)

    d = -2 * spin * expit(-z) / (M * N)

    grad_h = np.sum(d, axis=0)

    grad_J = d.T @ spin
    grad_J += grad_J.T

    # L2 regularization on off-diagonal couplings
    indices = np.triu_indices(N, k=1)

    loss += 0.5 * lambda_J * np.sum(J[indices]**2)
    grad_J += lambda_J * J

    # H1 regularization on fields
    h_grid = h.reshape(L, L)

    # Periodic nearest-neighbour differences
    diff_x = h_grid - np.roll(h_grid, 1, axis=0)
    diff_y = h_grid - np.roll(h_grid, 1, axis=1)

    reg_h = 0.5 * lambda_h * (
        np.sum(diff_x**2) + np.sum(diff_y**2)
    )

    loss += reg_h

    # Gradient of the H1 penalty
    grad_h_grid = lambda_h * (
        4 * h_grid
        - np.roll(h_grid, 1, axis=0)
        - np.roll(h_grid, -1, axis=0)
        - np.roll(h_grid, 1, axis=1)
        - np.roll(h_grid, -1, axis=1)
    )

    grad_h += grad_h_grid.ravel()

    # Extract independent upper-triangular couplings
    grad_couplings = grad_J[indices]

    gradient = np.concatenate((grad_h, grad_couplings))

    return loss, gradient


# --------------------------------------------------
# 5. MAIN
# --------------------------------------------------

def main():

    # Lattice dimensions.
    L = 8
    N = L * L

    # Read configurations.
    spin = read_spin_config("C:/Users/oscar/Desktop/SchoolRelated/ProjectCourseHT26/IsingModelProject/spin_config.txt", L)

    M = spin.shape[0]

    print("Number of configurations:", M)
    print("Number of spins:", N)

    # Calculate empirical observables.
    m_obs = avg_mag(spin)
    C_obs = avg_C(spin)

    print("Average magnetization:")
    print(m_obs)

    print("Average correlation matrix:")
    print(C_obs)

    # Number of unknown parameters.
    n_params = N + N * (N - 1) // 2

    # Initial guess for all parameters.
    theta0 = np.zeros(n_params)


    lambda_J = 1e-3
    lambda_h = 5e-1

    results = minimize(
        cost_func,
        theta0,
        args=(spin, L, lambda_J, lambda_h),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-10}
    )

    # Extract estimated parameters.
    J_est, h_est = unpack_parameters(results.x, N)

    print("\nOptimization successful:", results.success)
    print("Optimization message:", results.message)
    print("Final cost:", results.fun)

    print("\nEstimated magnetic fields:")
    print(h_est)

    print("\nEstimated coupling matrix:")
    print(J_est)

    plt.plot(h_est)
    plt.figure()
    plt.pcolormesh(range(N), range(N), J_est, shading='auto')
    plt.colorbar()
    plt.show()


    return J_est, h_est, results


if __name__ == "__main__":
    J_est, h_est, results = main()
