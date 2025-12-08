import numpy as np
import pandas as pd
from deepxde.metrics import mean_squared_error, l2_relative_error


def eval_predictions(real, pred):

    S_pred = pred[:,0]
    I_pred = pred[:,1]
    beta_pred = pred[:,2]

    S_real = real[:,0]
    I_real = real[:,1]
    beta_real = real[:,2]

    return pd.DataFrame({
        "compartiment": ["S", "I", "beta"], 
        "MSE": [
            mean_squared_error(S_real, S_pred),
            mean_squared_error(I_real, I_pred),
            mean_squared_error(beta_real, beta_pred)
        ],
        "L2": [
            l2_relative_error(S_real, S_pred),
            l2_relative_error(I_real, I_pred),
            l2_relative_error(beta_real, beta_pred)
        ],
        "L-infinity": [
            np.max(np.abs(S_real - S_pred)),
            np.max(np.abs(I_real - I_pred)),
            np.max(np.abs(beta_real - beta_pred))
        ]
    })