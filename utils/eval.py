import numpy as np
import pandas as pd
from deepxde.metrics import mean_squared_error, l2_relative_error


def eval_predictions(real, pred):
    S_pred = pred[:,0]
    I_pred = pred[:,1]
    beta_pred = pred[:,2]
