import numpy as np
import tensorflow as tf
import pandas as pd
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import minimize
from scipy.stats import linregress, poisson
from scipy.signal import savgol_filter

###############################################
############## beta(t) formats ################
###############################################
def mono_beta(t):
    # return t**2 / 10000 + 0.20
    return 0.25 * np.exp(-0.08 * t) + 0.2

def mono_beta_dfe(t):
    # return t**2 / 10000 + 0.25
    return 0.25 * np.exp(-0.08 * t) + 0.0

def sin_beta(t):
    return 0.1 * np.sin(0.44 * t) + 0.25

def sin_beta_dfe(t):
    return 0.1 * np.sin(0.44 * t) + 0.7

def sin_beta_szero(t):
    return 0.2 * np.sin(0.44 * t) + 0.45
###############################################


class RK4DataGenerator:


    def __init__(self, beta_t, gamma=0.1, t_0=0, t_f=50, N=100, S0=99, I0=1, R0=0):
        
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


def first_below_threshold(arr, threshold=1e-5):
    below_indices = np.where(arr < threshold)[0]
    if len(below_indices) > 0:
        index = below_indices[0]
        return arr[index], index
    return None, -1


def estimate_beta0_linregress(I_data, gamma, N, window):
    logI = np.log(I_data[:window])
    t = np.arange(len(logI))
    slope = linregress(t, logI)[0]
    return slope + gamma


def estimate_beta0_isaac(I_data, gamma, N, window):
    I0 = I_data[0]; I1 = I_data[1]; S0 = N - I0; deltat = 1
    beta0 = (N / (S0 * I0)) * ((I1 - I0) / (deltat) + gamma * I0)
    return beta0


def estimate_beta0_isaac2(I_data, gamma, N, window):
    I0 = I_data[0]; I1 = I_data[1]; I2 = I_data[2]; S0 = N - I0; deltat = 2
    beta0 = (N / (S0 * I0)) * ((-3*I0+4*I1 - I2) / (deltat) + gamma * I0)
    return beta0


def estimate_beta0_isaac3(I_data, gamma, N, window):
    I0 = I_data[0]; In = I_data[window]; S0 = N - I0
    return (N / (S0 * window)) * (np.log(In) - np.log(I0) + gamma * window)


def estimate_beta0_ode(I_data, gamma, N, window):
    
    def SIR_model(y, t, beta, gamma, N):
        S, I = y
        dS_dt = -beta[0] * S * I / N
        dI_dt = beta[0] * S * I / N - gamma * I
        return [dS_dt, dI_dt]

    def error(beta_guess, I_data, t, gamma, N):
        I0 = I_data[0]; S0 = N - I0; y0 = [S0, I0]
        solution = odeint(SIR_model, y0, t, args=(beta_guess, gamma, N))
        I_model = solution[:, 1]
        return np.sqrt(np.mean((np.log(I_model + 1) - np.log(I_data + 1))**2))

    result = minimize(
        error, 
        x0=0.5, 
        args=(I_data, np.arange(len(I_data)), gamma, N), 
        bounds=[(0.001, 2.0)]
    )

    return result.x[0]


def estimate_beta0_poisson(I_data, gamma, N, window):

    def log_likelihood(beta, I_data, gamma, N):
        n = len(I_data)
        S = np.zeros(n)
        I_model = np.zeros(n)
        
        S[0] = N - I_data[0]
        I_model[0] = I_data[0]
        
        logL = 0
        
        for i in range(n-1):
            expected_new = beta * S[i] * I_model[i] / N
            I_model[i+1] = I_model[i] + expected_new - gamma * I_model[i]
            S[i+1] = S[i] - expected_new
            observed_new = max(0, I_data[i+1] - I_data[i] + gamma * I_data[i])
            logL += poisson.logpmf(int(observed_new), expected_new)
        
        return -logL

    result = minimize(
        log_likelihood, 
        x0=0.5, 
        args=(I_data, gamma, N), 
        bounds=[(0.001, 2.0)]
    )
    
    return result.x[0]


def estimate_beta0_linear_aprox(I_data, gamma, N, window):
    n = len(I_data)
    t = np.arange(n)
    
    R = np.zeros(n)
    for i in range(1, n):
        R[i] = R[i-1] + gamma * (I_data[i-1] + I_data[i])/2
    
    S = N - I_data - R
    dI_dt = np.gradient(I_data, t)
    beta_t = (dI_dt/I_data + gamma) * N / S
    beta_t = beta_t[(S > 0.1*N) & (beta_t > 0) & (beta_t < 2)]
    
    return np.mean(beta_t)


def apply_savgol_filter(x, y, window=201, polyorder=3, index=None):
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    norm_x = (x - x_min) / (x_max - x_min)
    norm_y = (y - y_min) / (y_max - y_min)

    smooth_y = savgol_filter(
        norm_y, 
        window_length=window, 
        polyorder=polyorder
    )
    
    smooth_y = y_min + (y_max - y_min) * smooth_y

    if index is None:
        return smooth_y

    return pd.Series(smooth_y, index=index)