import numpy as np

def discretize(state):
    state[0] -= -1.2
    state[1] -= -0.07
    diff = [0.17, 0.014]
    state[0] /= diff[0]
    state[1] /= diff[1]
    state = np.round(state, 0).astype(int)
    return state