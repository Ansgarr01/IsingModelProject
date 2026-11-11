import numpy as np
from scipy.optimize import minimize


def avg_mag(spin):
    M = len(spin)
    tot = np.zeros_like(spin[0])
    for i in range(M):
        tot += spin[i]
    return tot/M

def avg_C(spin):
    M = len(spin)
    N = len(spin[0])
    C_obs = np.zeros_like(N, N)
    C_tot = np.zeros_like(N, N)
    for mu in range(0, M):
        sspin = spin[mu]
        for i in range(0, N):
            var = sspin[i]
            for j in range(i+1, N):
                const = var*sspin[j]
                C_obs[i, j] = const
                C_obs[j, i] = const
        C_tot += C_obs
    return C_tot/M ## C_avg


def cost_func():
    return

def main():

    ## READ FILE
    file = open("C:/Users/oscar/Desktop/SchoolRelated/ProjectCourseHT26/IsingModelProject/spin_config.txt", "r")
    data = file.read() ## Make 2D to 1D
    M = len(data)
    n = len(data[0])
    N = n*n
    spin = np.zeroes(M, N)
    for i in range(M):
        spin[i] = data[i].flatten()
    file.close()
    n_params = N + round(N*(N - 1)/2)
    theta = np.zeros(n_params)

    results = minimize(cost_func, )


        



