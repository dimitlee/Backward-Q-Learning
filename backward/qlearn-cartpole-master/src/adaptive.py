""" Q-Learning implementation for Cartpole """

import gym
import numpy as np
import collections
import math

env = gym.make('CartPole-v0')

# hyperparameters
buckets=(1, 1, 6, 12,)
n_episodes=1000
goal_duration=195
min_alpha=0.1  # learning rate
min_epsilon=0.1  # exploration rate
gamma_init=0.9  # discount factor
ada_divisor=25
temperature = 1.2
Q = np.zeros(buckets + (env.action_space.n,))

# helper functions
def discretize(obs):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)

def choose_action(state):
#    return env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(Q[state])
    temp = temperature
    pr_a = np.zeros(2, dtype=float)
    for i in range(2):
        pr_a[i] = math.exp(Q[state][i]/temp)
    sum_pr = np.sum(pr_a)
    pr_a = pr_a / sum_pr

    choose_action = np.random.random()
    prob = 0
    for i in range(2):
        prob += pr_a[i]
        if choose_action < prob :
            return i

def update_q(state_old, action, reward, state_new, alpha):
    Q[state_old][action] += alpha * (reward + gamma * np.max(Q[state_new]) - Q[state_old][action])

def get_epsilon(t):
    return max(min_epsilon, min(1, 1.0 - math.log10((t + 1) / ada_divisor)))

def get_alpha(t):
    return max(min_alpha, min(1.0, 1.0 - math.log10((t + 1) / ada_divisor)))


def run_episode(thn, thp, cc, bate, kek):
    """Run a single Q-Learning episode"""
    # get current state
    observation = env.reset()
    current_state = discretize(observation)

    # get learning rate and exploration rate
    alpha_init = 0.5
    alpha = alpha_init
    epsilon = get_epsilon(episode)
    gamma = 0.9
    done = False
    duration = 0
    iteration = 1
    # one episode of q learning
    while not done:
        # env.render()
        action = choose_action(current_state)
        obs, reward, done, _ = env.step(action)
        new_state = discretize(obs)
        temp = thn - kek*iteration
        temp2 = thp + kek*iteration 
        max_next_state_value = np.max(Q[new_state])
        if temp < max_next_state_value and temp2 > max_next_state_value :
            gamma = gamma_init
            alpha = alpha_init 
            temp3 = reward + gamma*max_next_state_value - Q[current_state][action]
        else :
            gamma_prev = gamma
            gamma = np.tanh(iteration*gamma_prev)
            temp3 = reward + gamma*max_next_state_value - Q[current_state][action]
            alpha = np.tanh(bate*abs(temp3)/iteration)
        temperature = cc/iteration
        update_q_value = Q[current_state][action] + alpha * temp3
        Q[current_state][action] = update_q_value
        current_state = new_state
        duration += 1
    
    return duration


def visualize_policy():
    """Visualize current Q-Learning policy without exploration / learning"""
    current_state = discretize(env.reset())
    done=False

    while not done:
        action = choose_action(current_state, 0)
        obs, reward, done, _ = env.step(action)
        env.render()
        current_state = discretize(obs)

    env.close()

    return


if __name__ == '__main__':
    durations = collections.deque(maxlen=100)
    print(env.action_space)
    th_n = 800
    th_p = -600
    c = 1.2
    beta = 2
    k = 1.15
    for episode in range(n_episodes):
        duration = run_episode(th_n, th_p, c, beta, k)
        
        # mean duration of last 100 episodes
        durations.append(duration)
        mean_duration = np.mean(durations)

        # check if our policy is good
        if mean_duration >= goal_duration and episode >= 100:
            print('Ran {} episodes. Solved after {} trials'.format(episode, episode - 100))
            visualize_policy()
            break
        
        elif episode % 100 == 0:
            print('[Episode {}] - Mean time over last 100 episodes was {} frames.'.format(episode, mean_duration))
