import numpy as np
import math

from const import TEMP, TEMP_MIN, VLOW, LOW, HIGH
from mount_car_back_env import MountCarBackEnv

env = MountCarBackEnv()

def boltz_policy(state1, state2, q_table, temp=TEMP, debug=False):
    """
    Boltzmann distribution policy determines probability distribution of actions according
    to Boltzmann distribution.
    P(a_i) = exp(Q(s, a_i)/temp) / (sum of exp(Q/temp) over all actions)
    """
    # Store probabilities of each action in an array
    n = env.action_space.n
    pr_a = np.zeros(n, dtype=float)
    for i in range(n):
        pr_a[i] = math.exp(q_table[state1, state2, i]/temp)
    sum_pr = np.sum(pr_a)
    if(sum_pr == 0):
        return np.random.choice(n)
    pr_a = pr_a / sum_pr
    # print(pr_a)

    choose_action = np.random.random()
    prob = 0
    action = -1
    for i in range(n):
        prob += pr_a[i]
        if choose_action < prob :
            action = i
    if action == -1:
        action = n - 1
    return action