import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp


###############################################
############## beta(t) formats ################
###############################################
def mono_beta(t):
    return t**2 / 10000 + 0.20

def mono_beta_dfe(t):
    return t**2 / 10000 + 0.25

def sin_beta(t):
    return 0.1 * np.sin(0.44 * t) + 0.25

def sin_beta_dfe(t):
    return 0.1 * np.sin(0.44 * t) + 0.25
###############################################


class RK4DataGenerator:


    def __init__(self, beta_t, gamma=0.1, t_0=0, t_f=50, N=1, S0=0.99, I0=0.01, R0=0.0):
        
        def sir_system(t, comparts, beta_t, gamma):
            S, I, R = comparts
            return [
                -beta_t(t) * S * I / N,
                beta_t(t) * S * I / N - gamma * I,
                gamma * I
            ]
        
        self.t_0 = t_0
        self.t_f = t_f
        self.N = N
        self.sir_sol = solve_ivp(
            sir_system,
            [t_0, t_f],
            [S0, I0, R0],
            args=[beta_t, gamma],
            dense_output=True
        )


    def generate(self, t, noise_std=0.0):
        comparts = self.sir_sol.sol(t).T
        comparts += np.random.normal(scale=noise_std, size=(len(t), 3))
        comparts[comparts < 0] = 0
        comparts[comparts > self.N] = self.N
        return comparts


def stack_real_values(sir_real, beta_real, n_compartments=2):
    comparts = sir_real[:,0:n_compartments]
    return np.vstack((comparts.T, beta_real)).T


def transform_column_with_mask(data, column_idx, transform_fn):
    col_mask = tf.one_hot(column_idx, depth=data.shape[1], dtype=data.dtype)
    row_mask = tf.ones([data.shape[0], 1], dtype=data.dtype)
    mask = row_mask @ tf.reshape(col_mask, [1, -1])
    
    transformed_data = transform_fn(data)
    result = mask * transformed_data + (1 - mask) * data

    return result
