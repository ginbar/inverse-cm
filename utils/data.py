import numpy as np
from scipy.integrate import solve_ivp


###############################################
############## beta(t) formats ################
###############################################
def mono_beta(t):
    return t**2 / 10000 + 0.25

def mono_beta_dfe(t):
    return t**2 / 10000 + 0.25

def sin_beta(t):
    return 0.1 * np.sin(0.44 * t) + 0.25

def sin_beta_dfe(t):
    return 0.1 * np.sin(0.44 * t) + 0.25
###############################################


class RK4DataGenerator:


    def __init__(self, beta_t, gamma=0.1, t_0=0, t_f=50, S0=0.99, I0=0.01, R0=0.0):
        
        def sir_system(t, comparts, gamma):
            S, I, R = comparts
            return [
                -beta_t(t) * S * I,
                beta_t(t) * S * I - gamma * I,
                gamma * I
            ]
        
        self.t_0 = t_0
        self.t_f = t_f
        self.sir_sol = solve_ivp(
            sir_system,
            [t_0, t_f],
            [S0, I0, R0],
            args=[gamma],
            dense_output=True
        )


    def generate(self, t, noise_std=0.0):
        comparts = self.sir_sol.sol(t).T
        comparts += np.random.normal(scale=noise_std, size=(len(t), 3))
        comparts[comparts < 0] = 0
        comparts[comparts > 1] = 1
        return comparts
