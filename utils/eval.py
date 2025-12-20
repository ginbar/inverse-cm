import numpy as np
import pandas as pd
from deepxde.metrics import mean_squared_error, l2_relative_error


def eval_predictions(real, pred, compartiments=["S", "I", "beta"]):

    indexes = range(len(compartiments))

    return pd.DataFrame({
        "compartiment": compartiments, 
        "MSE": [mean_squared_error(real[:,i], pred[:,i]) for i in indexes],
        "L2": [l2_relative_error(real[:,i], pred[:,i]) for i in indexes],
        "L-infinity": [np.max(np.abs(real[:,i] - pred[:,i])) for i in indexes]
    })