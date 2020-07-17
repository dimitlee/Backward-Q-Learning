""" Q-Learning implementation for Cartpole """

import gym
import numpy as np
import collections
import math
from math import floor
env = gym.make('CartPole-v0')

# hyperparameters
buckets1=(3, 3, 6, 3,)
buckets2 = [3,3,6,3]
low = env.observation_space.low
high = env.observation_space.high
n_episodes=5000
goal_duration=120000
min_alpha=0.1  # learning rate
min_epsilon=0.1  # exploration rate
gamma=0.99  # discount factor
ada_divisor=25
Q = np.zeros(buckets1 + (env.action_space.n,))
low[1] = -0.5
low[3] = -math.radians(50)
high[1] = 0.5
high[3] = math.radians(50)
dim = len(low) 
width = []        # width of each quantization step
for idx in range(dim):
    width.append((high[idx] - low[idx]) / buckets2[idx]) 
# helper functions
def quantize(observation):
    quantized_obs = []
    for idx in range(dim):
        if observation[idx] < low[idx]:
            quantized_obs.append(0)
        elif observation[idx] >= high[idx]:
            quantized_obs.append(buckets2[idx]-1)
        else:
            quantized_obs.append(int(floor((observation[idx] - low[idx])/ width[idx])))
    return tuple(quantized_obs)
def discretize(obs):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)

def choose_action(state, tem):
#    return env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(Q[state])
    temp = tem
    pr_a = np.zeros(5, dtype=float)
    for i in range(5):
        pr_a[i] = math.exp(Q[state][i]/temp)
    sum_pr = np.sum(pr_a)
    pr_a = pr_a / sum_pr

    choose_action = np.random.random()
    prob = 0
    for i in range(5):
        prob += pr_a[i]
        if choose_action < prob :
            return i

def update_q(state_old, action, reward, state_new, alpha, tempe):
    Q[state_old][action] += alpha * (reward + gamma * np.max(Q[state_new]) - Q[state_old][action])

def get_epsilon(t):
    return max(min_epsilon, min(1, 1.0 - math.log10((t + 1) / ada_divisor)))

def get_alpha(t):
    return max(min_alpha, min(1.0, 1.0 - math.log10((t + 1) / ada_divisor)))


def run_episode(tempr):
    """Run a single Q-Learning episode"""
    # get current state
    observation = env.reset()
    current_state = quantize(observation)

    # get learning rate and exploration rate
    alpha = 0.1
    epsilon = get_epsilon(episode)

    done = False
    duration = 0

    # one episode of q learning
    while not done:
        #env.render()
        action = choose_action(current_state, tempr)
        obs, reward, done, _ = env.step(action)
        new_state = quantize(obs)
        update_q(current_state, action, reward, new_state, alpha, tempr)
        current_state = new_state
        duration += 1 
    return duration


def visualize_policy(tempr2):
    """Visualize current Q-Learning policy without exploration / learning"""
    current_state = quantize(env.reset())
    done=False

    while not done:
        action = choose_action(current_state, tempr2)
        obs, reward, done, _ = env.step(action)
        env.render()
        current_state =quantize(obs)

    env.close()

    return


if __name__ == '__main__':
    durations = collections.deque(maxlen=5000)
    print(env.action_space)
    print(env.observation_space.shape[0])
    temper = 0.06
    for episode in range(n_episodes):
        duration = run_episode(temper)
        #temper = 0.99937*temper     
        #print(temper)   
        # mean duration of last 100 episodes
        durations.append(duration)
        mean_duration = np.mean(durations)
        sum = np.sum(durations)

        # check if our policy is good
        if sum>=goal_duration:
            print('[Episode {}] - Total time over last episodes was {} frames.'.format(episode, sum))
            print("with mean result:")
            print(mean_duration)
            print(temper)            
            visualize_policy(temper)
            break
        
        elif episode % 100 == 0:
            print('[Episode {}] - Sum time over all episodes was {} frames.'.format(episode, sum))

   
    env.close()
