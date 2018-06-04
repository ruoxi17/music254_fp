import numpy as np


# deterministic 
def argmax(array):
    return np.argmax(array)


# stochastic; higher tau: add variety; default: 1, raw proba
def temperature(tau=1):
    
    def ret_fun(array):
        proba = np.power(array, 1.0 / tau)
        proba = proba / np.sum(proba)
        idx = np.random.choice(len(array), 1, p=proba)[0]
        return idx
    
    return ret_fun