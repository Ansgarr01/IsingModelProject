import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe


def local_H(spin, i, j, K):
    L = spin.shape[0]
    
    up = spin[(i-1) % L, j]
    down = spin[(i+1)% L, j]
    left = spin[i, (j-1) % L]
    right = spin[i, (j+1) % L]
    s_ij = spin[i,j] 
    
    H = -K*s_ij*(up+down+left+right)
    return H

def total_H(spin, K, B):
    L = spin.shape[0]
    total_energy = 0
    for i in range(L):
        for j in range(L):
            total_energy += local_H(spin, i, j , K)
    total_energy /= 2
    total_energy -= B*np.sum(spin)
    return total_energy


def metropolis(spin,K,B):
    L = spin.shape[0]
    N = L*L
    N = int(N)
    a = 0
    for _ in range(N):
        i = np.random.randint(0,L)
        j = np.random.randint(0,L)
        E_before = local_H(spin,i,j,K)
        s = spin[i,j]
        # After
        spin[i,j] *=-1
        E_after = local_H(spin,i,j,K)

        dE = E_after - E_before + 2*B*s
        w = np.exp(-dE)
        u = np.random.rand()
        if dE <= 0 or u < w:
            a += 1 # Accept the move
        else:
            spin[i,j] *= -1 # else flip back
    return spin, a /N

def heat_bath(spin,K,B):
    L = spin.shape[0]
    N = L*L
    N = int(N)
    a = 0
    for _ in range(N):
        i = np.random.randint(0,L)
        j = np.random.randint(0,L)
        up = spin[(i-1) % L, j]
        down = spin[(i+1)% L, j]
        left = spin[i, (j-1) % L]
        right = spin[i, (j+1) % L]
        S_ij = up + down + left + right
        
        p = np.exp(2*K*(S_ij))/(1 + np.exp(2*K*S_ij))
        u = np.random.rand()
        if p > u:
            spin[i,j] = 1
            a += 1
        else:
            spin[i,j] = -1
    return spin, a / N
      

def observables(spin, K, B):
    M = np.sum(spin)
    E = total_H(spin,K,B)
    M_staggered = staggered_magnetization(spin)
    
    return M, E, M_staggered
    
def monte_carlo(spin, K , n_therm , n_measure , interval , B):
    # Start running the n_therm iterations
    a_list = []
    E_test = []
    staggered_list = []
    file = open("spin_config.txt", "w+")
    spin_interval = 50
    for _ in range (n_therm):
        spin , a = metropolis(spin,K,B)
        M, E, staggered = observables(spin,K,B)
        E_test.append(E)
    
    # n_measure steps to take measurements
    M_list = []
    E_list = []
    for m in range (n_measure):
        spin, a = metropolis(spin,K,B)
        a_list.append(a)
        if m % interval == 0:
            M, E, staggered = observables(spin,K,B)
            M_list.append(M)
            E_list.append(E)
            E_test.append(E)
            if m % spin_interval == 0:
                staggered_list.append(staggered)
                file.write(str(spin))
    file.close()
            
    return np.array(E_list), np.array(M_list), a_list, np.array(E_test), np.array(staggered_list)

def monte_carlo_heat_bath(spin, K , n_therm , n_measure , interval , B):
    # Start running the n_therm iterations
    a_list = []
    E_test = []
    staggered_list = []
    for _ in range (n_therm):
        spin , a = heat_bath(spin,K,B)
        M, E, staggered = observables(spin,K,B)
        E_test.append(E)
    
    # n_measure steps to take measurements
    M_list = []
    E_list = []
    for m in range (n_measure):
        spin, a = heat_bath(spin,K,B)
        a_list.append(a)
        if m % interval == 0:
            M, E, staggered = observables(spin,K,B)
            staggered_list.append(staggered)
            M_list.append(M)
            E_list.append(E)
            E_test.append(E)
            
    return np.array(E_list), np.array(M_list), a_list, np.array(E_test), np.array(staggered_list)
            
def staggered_magnetization(spin):
    L = spin.shape[0]
    N = L * L
    M_s = 0.0
    
    for i in range(L):
        for j in range(L):
            if (i + j) % 2 == 0:
                M_s += spin[i, j]
            else:
                M_s -= spin[i, j]
    
    return M_s / N
    

def M_exact(T):
    T = np.array(T)
    M = np.zeros_like(T)
    
    for i, t in enumerate(T):
        if t < 2.3:   # T_c
            z = np.exp(-2/t)
            M[i] = (( (1 + z**2)**0.5 * (1 - 6*z**2 + z**4) )
                     / (1 - z**2)**2 )**(1/8)
        else:
            M[i] = 0.0

    return M

def kappa(T):
    K = 1/T
    return 2*np.sinh(2*K)/(np.cosh(2*K)**2)

def kappa_prime(T):
    K = 1/T
    return 2*np.tanh(2*K)**2-1

def C_exact(T):
    kap = kappa(T)
    kap_p = kappa_prime(T)
    K = 1/T
    k1 = ellipk(kap**2)
    e1 = ellipe(kap**2)
    C = 2/np.pi*(K*np.cosh(2*K)/np.sinh(2*K))**2*(2*k1 - 2*e1 - (1-kap_p)* (np.pi/2 + kap_p*k1))
    return C

def E_exact(T):
    K = 1/T
    kap = kappa(T)
    
    k1 = ellipk(kap**2)
    E =  -K*(np.cosh(2*K)/np.sinh(2*K))* (1 + (2/np.pi)*kappa_prime(T)*k1)
    return E

def run_simulation(L, n_therm, n_measure, interval, B):
#     T = np.linspace(0.5, 5, 20)
    T = 2
    J = -1
    M = []
    E = []
    chi_list = []
    s_heat_list = []
    a_list = []
    cumulant_list = []
    metropolis_tests = []
    staggered_list = []
    
    # Heat bath
    M_hb = []
    cumulant_list_hb = []
    E_hb = []
    chi_list_hb = []
    s_heat_list_hb = []
    heat_bath_tests = []
    
    counter = 0
    i = T
    print(f'----------- Size = {L} x {L} -----------')
    K = J / i
    P = B / i
    spin = np.random.choice([-1,1], size = (L,L))
    E_list, M_list, a, metropolis_test, staggered = monte_carlo(spin, K, n_therm, n_measure, interval, P)
    #Heat Bath Monte Carlo
    E_list_hb, M_list_hb, a_hb, heat_bath_test, hb_staggered = monte_carlo_heat_bath(spin, K, n_therm, n_measure, interval,B)
    # Sample the thermalization at T = Tc
    N = L*L
    # ------- Metropolis ---------
    # Magnetization
    M_avg = np.mean(M_list) / N
    M.append(np.abs(M_avg))
    M2_avg = np.mean(M_list**2) / N**2
    M4_avg = np.mean(M_list**4) / N**4
    
    #Staggered Magnetization
    staggered_list.append(np.mean(np.abs(staggered)))
    
    #Cumulant
    cumulant = 1 - M4_avg/(3*M2_avg**2)
    cumulant_list.append(cumulant)
    

    # Energy 
    E_avg = np.mean(E_list)
    E2_avg = np.mean(E_list**2)

    E.append(E_avg / N)

    # Susceptibility
    chi = N*(M2_avg - M_avg**2)
    chi_list.append(chi)

    # Specific heat 
    s_heat = (E2_avg - E_avg**2)/N
    s_heat_list.append(s_heat)
    
    #-------Heat Bath----------
    # Magnetization
    M_avg_hb = np.mean(M_list_hb) / N
    M2_avg_hb = np.mean(M_list_hb**2) / N**2
    M4_avg_hb = np.mean(M_list_hb**4) / N**4
    M_hb.append(np.abs(M_avg_hb))
    
    #Cumulant
    cumulant_hb = 1 - M4_avg_hb/(3*M2_avg_hb**2)
    cumulant_list_hb.append(cumulant_hb)
    
    # Energy 
    E_avg_hb = np.mean(E_list_hb)
    E2_avg_hb = np.mean(E_list_hb**2)

    E_hb.append(E_avg_hb / N)

    # Susceptibility
    chi_hb = N*(M2_avg_hb - M_avg_hb**2)
    chi_list_hb.append(chi_hb)

    # Specific heat 
    s_heat_hb = (E2_avg_hb - E_avg_hb**2)/N
    s_heat_list_hb.append(s_heat_hb)
    
    a_list.append(float(np.mean(a)))
        
    
    print(f'Acceptance rate: {np.round(a_list,4)}')
    print(f'Acceptance rate: {np.round(np.mean(a_list),4)}')
#     M_inf = M_exact(T)
#     E_inf = E_exact(T)
#     C_inf = C_exact(T)
    
    
#     plt.figure()
#     plt.plot(T,staggered_list, label = 'Metropolis')
#     plt.plot(T,M_inf, label = 'Analytical')
#     plt.plot(T,M_hb, label = 'Heat Bath')
#     plt.legend()
#     plt.xlabel('T')
#     plt.title(f'Staggered Magnetization, size = {L} x {L}')
#     
#     plt.figure()
#     plt.plot(metropolis_tests, label = 'Metropolis')
#     plt.plot(heat_bath_tests, label = 'Heat Bath')
#     plt.legend()
#     plt.xlabel('Iterations')
#     plt.title('Thermalization approximation')
#     
#     plt.figure()
#     plt.plot(T,M, label = 'Metropolis')
#     plt.plot(T,M_inf, label = 'Analytical')
#     plt.plot(T,M_hb, label = 'Heat Bath')
#     plt.legend()
#     plt.xlabel('T')
#     plt.title(f'M vs T, size = {L} x {L}')
#     
#     plt.figure()
#     plt.plot(T,E, label = 'Metropolis')
#     plt.plot(T,E_inf, label = 'Analytical')
#     plt.plot(T,E_hb, label = 'Heat Bath')
#     plt.legend()
#     plt.xlabel('T')
#     plt.title(f'E vs T, size = {L} x {L}')
#     
#     plt.figure()    
#     plt.plot(T,chi_list, label = 'Metropolis')
#     plt.plot(T,chi_list_hb, label = 'Heat Bath')
#     plt.legend()
#     plt.xlabel('T')
#     plt.title(fr' $\chi$ vs T, size = {L} x {L}')
#     
#     plt.figure()    
#     plt.plot(T,s_heat_list, label = 'Metropolis')
#     plt.plot(T,C_inf, label = 'Analytical')
#     plt.plot(T,s_heat_list_hb, label = 'Heat Bath')
#     plt.legend()
#     plt.xlabel('T')
#     plt.title(f'C vs T, size = {L} x {L}')
    
    return 0
    

n_therm = 2000
n_measure = 5000
interval = 5
B = 0

#T = np.linspace(0.5,10,20)
T = 2
binder_cumulant = []
L = [8]
for i in L:
    cumulant = run_simulation(i, n_therm, n_measure, interval, B)

plt.show()
