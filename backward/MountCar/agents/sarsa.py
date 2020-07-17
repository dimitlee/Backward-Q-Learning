import numpy as np

from agents.policy import boltz_policy
from agents.discretizer import discretize
from const import TEMP, TEMP_MIN, VLOW, LOW, HIGH

def sarsa(env, learn_st, test_st, learning=0.5, discount=0.99, back=False):
    # Initialize Q table
    Q = np.random.uniform(low = -1, high = 1, 
                          size = (11, 11, 
                                  env.action_space.n))
    
    learning_b = 0.05
    discount_b = 0.99

    # Initialize variables to track results
    learn_len = 0
    learn_cumr = 0
    tot_episodes = 0

    # Decrease temp every turn
    temp = TEMP
    decay = (TEMP - TEMP_MIN) / 5000
    
    reached = 0
    # Run Q learning algorithm
    while reached < 300:
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        x = np.random.choice(300)
        pos, vel = learn_st[x]
        state = env.reset(pos, vel)
        M = list()
        steps = 0
        
        # Discretize state
        state_adj = discretize(state)
        state_tup = (state_adj[0], state_adj[1])

        # Determine action - Boltzmann policy
        action = boltz_policy(state_adj[0], state_adj[1], Q, temp)
    
        while done != True:
            # Get next state and reward
            state2, reward, done, info = env.step(action)
            steps += 1
            
            # Discretize state2
            state2_adj = discretize(state2)
            state2_tup = (state2_adj[0], state2_adj[1])
            
            # Determine action in state2 - Boltzmann policy
            action2 = boltz_policy(state2_adj[0], state2_adj[1], Q, temp)

            # Store s, a, r, s' in M
            if back:
                M.append((state_tup, action, reward, state2_tup))

            if done and reward == 200:
                reached += 1
                
            # Adjust Q value for current state
            else:
                delta = learning*(reward + 
                                 discount*Q[state2_adj[0], state2_adj[1], action2] - 
                                 Q[state_adj[0], state_adj[1], action])
                Q[state_adj[0], state_adj[1],action] += delta

            state_adj = state2_adj
            action = action2

            # Update variables
            tot_reward += reward
            state_adj = state2_adj

        while back and tot_reward > -2500 and steps > 0:
            state1, action, reward, state2 = M.pop()
            state11, state12 = state1
            state21, state22 = state2
            Q[state11, state12, action] += learning_b*(reward +
                discount_b*max(Q[state21, state22]) -
                Q[state11, state12, action])
            steps -= 1

        if back:
            M.clear()

        if temp > TEMP_MIN:
            temp -= decay
        
        tot_episodes += 1

        # Track mean length of episodes and mean cumulative reward
        learn_len = (learn_len*(tot_episodes - 1) + steps)/tot_episodes
        learn_cumr = (learn_cumr*(tot_episodes - 1) + tot_reward)/tot_episodes

    test_len = 0
    test_cumr = 0
    tot_episodes = 0
    reached = 0
    # Run SARSA algorithm: Testing phase
    while reached < 40:
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        pos, vel = test_st[reached]
        state = env.reset(pos, vel)
        M = list()
        steps = 0
        
        # Discretize state
        state_adj = discretize(state)
        state_tup = (state_adj[0], state_adj[1])

        # Determine action - Boltzmann policy
        action = boltz_policy(state_adj[0], state_adj[1], Q, temp)
    
        while done != True:
            # Get next state and reward
            state2, reward, done, info = env.step(action)
            steps += 1
            
            # Discretize state2
            state2_adj = discretize(state2)
            state2_tup = (state2_adj[0], state2_adj[1])
            
            # Determine action in state2 - Boltzmann policy
            action2 = boltz_policy(state2_adj[0], state2_adj[1], Q, temp)

            # Store s, a, r, s' in M
            if back:
                M.append((state_tup, action, reward, state2_tup))

            if done:
                reached += 1
                
            # Adjust Q value for current state
            else:
                delta = learning*(reward + 
                                 discount*Q[state2_adj[0], state2_adj[1], action2] - 
                                 Q[state_adj[0], state_adj[1], action])
                Q[state_adj[0], state_adj[1],action] += delta

            state_adj = state2_adj
            action = action2

            # Update variables
            tot_reward += reward
            state_adj = state2_adj

        while back and tot_reward > -2500 and steps > 0:
            state1, action, reward, state2 = M.pop()
            state11, state12 = state1
            state21, state22 = state2
            Q[state11, state12, action] += learning_b*(reward +
                discount_b*max(Q[state21, state22]) -
                Q[state11, state12, action])
            steps -= 1

        if back:
            M.clear()

        if temp > TEMP_MIN:
            temp -= decay
        
        tot_episodes += 1

        # Track mean length of episodes and mean cumulative reward
        test_len = (test_len*(tot_episodes - 1) + steps)/tot_episodes
        test_cumr = (test_cumr*(tot_episodes - 1) + tot_reward)/tot_episodes
            
    env.close()
    
    return learn_len, learn_cumr, test_len, test_cumr