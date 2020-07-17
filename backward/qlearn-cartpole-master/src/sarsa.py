""" Q-Learning implementation for Cartpole """

import gym
import numpy as np
import collections
import math

env = gym.make('CartPole-v0')

# hyperparameters
buckets=(1, 1, 6, 12,)
n_episodes=5000
goal_duration=120000
min_alpha=0.1  # learning rate
min_epsilon=0.1  # exploration rate
gamma=1.0  # discount factor
ada_divisor=25
Q = np.zeros(buckets + (env.action_space.n,))

# helper functions
def discretize(obs):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)

def choose_action(state, tem):
#    return en.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(Q[state])
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
    next_action = choose_action(state_new, tempe)
    Q[state_old][action] += alpha * (reward + gamma * Q[state_new][next_action] - Q[state_old][action])
    return next_action

def get_epsilon(t):
    return max(min_epsilon, min(1, 1.0 - math.log10((t + 1) / ada_divisor)))

def get_alpha(t):
    return max(min_alpha, min(1.0, 1.0 - math.log10((t + 1) / ada_divisor)))


def run_episode(tempr):
    """Run a single Q-Learning episode"""
    # get current state
    observation = env.reset()
    current_state = discretize(observation)

    # get learning rate and exploration rate
    alpha = 0.1
    epsilon = get_epsilon(episode)

    done = False
    duration = 0
    action = choose_action(current_state, tempr)
    # one episode of q learning
    while not done:
        # env.render()
        obs, reward, done, _ = env.step(action)
        new_state = discretize(obs)
        next_act = update_q(current_state, action, reward, new_state, alpha, tempr)
        current_state = new_state
        duration += 1
        action = next_act
    return duration


def visualize_policy(tempr2):
    """Visualize current Q-Learning policy without exploration / learning"""
    current_state = discretize(env.reset())
    done=False

    while not done:
        action = choose_action(current_state, tempr2)
        obs, reward, done, _ = env.step(action)
        env.render()
        current_state = discretize(obs)

    env.close()

    return


if __name__ == '__main__':
    durations = collections.deque(maxlen=5000)
    print(env.action_space)
    temper = 8.79
    for episode in range(n_episodes):
        duration = run_episode(temper)
        temper = 0.999*temper
        # mean duration of last 100 episodes
        durations.append(duration)
        mean_duration = np.mean(durations)
        summ = np.sum(durations)
        # check if our policy is good
        if summ>=goal_duration:
            print('[Episode {}] - Total time over last episodes was {} frames.'.format(episode, summ))
            print("with mean result:")
            print(mean_duration)
            visualize_policy(temper)
            break
        elif episode % 100 == 0:
            print('[Episode {}] - Mean time over last 100 episodes was {} frames.'.format(episode, summ))
    env.close()
