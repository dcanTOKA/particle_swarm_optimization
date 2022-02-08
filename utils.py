import numpy as np


def objective_function(var):
    # maximize the function below :
    # func = np.cos(var) + np.sin(var)
    # func = 1 + 2 * var - np.square(var)
    # func = 5 + 4 * var - np.square(var)
    func = 8 + 6 * var + np.square(var)
    return func
