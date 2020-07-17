import numpy as np
import gym
import matplotlib.pyplot as plt
import math
from numpy.random import default_rng

from mount_car_back_env import MountCarBackEnv
from agents.QLearning import QLearning
from agents.sarsa import sarsa
from agents.sa_q_learning import sa_q_learning
from const import TEMP, TEMP_MIN, VLOW, LOW, HIGH
from states import learn_st, train_st

# Import and initialize Mountain Car Environment
env = MountCarBackEnv()
env.reset()

results = open("results.txt", "w+")
# Run the experiment
# Q-Learning
results.write("Q-Learning\n")
avg_le, avg_lcum, avg_tle, avg_tcum = 0, 0, 0, 0
for i in range(100):
    le, lcum, tle, tcum = QLearning(env, learn_st, train_st)
    avg_le = (avg_le*i + le)/(i + 1)
    avg_lcum = (avg_lcum*i + lcum)/(i + 1)
    avg_tle = (avg_tle*i + tle)/(i + 1)
    avg_tcum = (avg_tcum*i + tcum)/(i + 1)
results.write(f"LE: {avg_le}\n")
results.write(f"L Cum: {avg_lcum}\n")
results.write(f"TLE: {avg_tle}\n")
results.write(f"T Cum: {avg_tcum}\n")

# Q-Learning - back
results.write("Q-Learning\n")
avg_le, avg_lcum, avg_tle, avg_tcum = 0, 0, 0, 0
for i in range(100):
    le, lcum, tle, tcum = QLearning(env, learn_st, train_st, back=True)
    avg_le = (avg_le*i + le)/(i + 1)
    avg_lcum = (avg_lcum*i + lcum)/(i + 1)
    avg_tle = (avg_tle*i + tle)/(i + 1)
    avg_tcum = (avg_tcum*i + tcum)/(i + 1)
results.write(f"LE: {avg_le}\n")
results.write(f"L Cum: {avg_lcum}\n")
results.write(f"TLE: {avg_tle}\n")
results.write(f"T Cum: {avg_tcum}\n")

# SARSA
results.write("SARSA\n")
avg_le, avg_lcum, avg_tle, avg_tcum = 0, 0, 0, 0
for i in range(100):
    le, lcum, tle, tcum = sarsa(env, learn_st, train_st)
    avg_le = (avg_le*i + le)/(i + 1)
    avg_lcum = (avg_lcum*i + lcum)/(i + 1)
    avg_tle = (avg_tle*i + tle)/(i + 1)
    avg_tcum = (avg_tcum*i + tcum)/(i + 1)
results.write(f"LE: {avg_le}\n")
results.write(f"L Cum: {avg_lcum}\n")
results.write(f"TLE: {avg_tle}\n")
results.write(f"T Cum: {avg_tcum}\n")

# SARSA - back
results.write("SARSA\n")
avg_le, avg_lcum, avg_tle, avg_tcum = 0, 0, 0, 0
for i in range(100):
    le, lcum, tle, tcum = sarsa(env, learn_st, train_st, back=True)
    avg_le = (avg_le*i + le)/(i + 1)
    avg_lcum = (avg_lcum*i + lcum)/(i + 1)
    avg_tle = (avg_tle*i + tle)/(i + 1)
    avg_tcum = (avg_tcum*i + tcum)/(i + 1)
results.write(f"LE: {avg_le}\n")
results.write(f"L Cum: {avg_lcum}\n")
results.write(f"TLE: {avg_tle}\n")
results.write(f"T Cum: {avg_tcum}\n")

# SA-Q-Learning
results.write("SA-Q-Learning\n")
avg_le, avg_lcum, avg_tle, avg_tcum = 0, 0, 0, 0
for i in range(100):
    le, lcum, tle, tcum = sa_q_learning(env, learn_st, train_st)
    avg_le = (avg_le*i + le)/(i + 1)
    avg_lcum = (avg_lcum*i + lcum)/(i + 1)
    avg_tle = (avg_tle*i + tle)/(i + 1)
    avg_tcum = (avg_tcum*i + tcum)/(i + 1)
results.write(f"LE: {avg_le}\n")
results.write(f"L Cum: {avg_lcum}\n")
results.write(f"TLE: {avg_tle}\n")
results.write(f"T Cum: {avg_tcum}\n")

# SA-Q-Learning - back
results.write("SA-Q-Learning\n")
avg_le, avg_lcum, avg_tle, avg_tcum = 0, 0, 0, 0
for i in range(100):
    le, lcum, tle, tcum = sa_q_learning(env, learn_st, train_st, back=True)
    avg_le = (avg_le*i + le)/(i + 1)
    avg_lcum = (avg_lcum*i + lcum)/(i + 1)
    avg_tle = (avg_tle*i + tle)/(i + 1)
    avg_tcum = (avg_tcum*i + tcum)/(i + 1)
results.write(f"LE: {avg_le}\n")
results.write(f"L Cum: {avg_lcum}\n")
results.write(f"TLE: {avg_tle}\n")
results.write(f"T Cum: {avg_tcum}\n")
  