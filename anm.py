"""
ANM baseline wrapper for bivariate causal discovery experiments.

This implementation uses the Additive Noise Model (ANM) implementation provided by:
causal-learn: https://github.com/py-why/causal-learn

License: MIT License
"""
import numpy as np
from causallearn.search.FCMBased.ANM.ANM import ANM

def predict_causal_direction_anm(x, y):

    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-8)
    y_norm = (y - np.mean(y)) / (np.std(y) + 1e-8)

    x_norm = x_norm.reshape(-1, 1)
    y_norm = y_norm.reshape(-1, 1)

    anm = ANM()
    
    pval_x2y, pval_y2x = anm.cause_or_effect(x_norm, y_norm)

    if pval_x2y > pval_y2x:
        return 1  # X -> Y
    else:
        return -1 # Y -> X