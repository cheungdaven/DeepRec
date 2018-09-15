import numpy as np


def RMSE(error, num):
    return np.sqrt(error / num)


def MAE(error_mae, num):
    return (error_mae / num)
