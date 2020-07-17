import numpy as np

from agents.policy import boltz_policy
from agents.discretizer import discretize
from const import TEMP, TEMP_MIN, VLOW, LOW, HIGH

def adaptive(env, learn_st, test_st, learning=0.5, discount=0.99, temp = TEMP, back=False):
    # Initialize Q table
    Q = np.random.uniform(low = -1, high = 1, 
                          size = (11, 11, 
                                  env.action_space.n))
    
    learning_b = 0.05
    discount_b = 0.99
    th_p = 100
    th_n = -100
    c = 50
    beta = 2
    k = 1.15
    discount_init = discount
    learning_init = learning
    # Initialize variables to track results
    learn_len = 0
    learn_cumr = 0
    tot_episodes = 0
    
    reached = 0
    # Run Q learning algorithm
    while reached < 300
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        x = np.random.choice(300)
        pos, vel = learn_st[x]
        state = env.reset(pos, vel)
        steps = 0
        M = list()
        
        # Discretize state
        state_adj = discretize(state)
        state_tup = (state_adj[0], state_adj[1])
    
        while done != True:   
            # Determine next action - Boltzmann policy
            action = boltz_policy(state_adj[0], state_adj[1], Q, temp)
                
            # Get next state and reward
            state2, reward, done, info = env.step(action)
            steps += 1 
            
            # Discretize state2
            state2_adj = discretize(state2)
            state2_tup = (state2_adj[0], state2_adj[1])

            # Store s, a, r, s' in M
            if back:
                M.append((state_tup, action, reward, state2_tup))
            
            if done and reward == 200:
                reached += 1

            max_next = max(Q[state2_adj[0], state2_adj[1]])

            if ((th_n - k*steps) < max_next) and ((th_p + k*steps) > max_next) :
                discount = discount_init
                learning = learning_init
            else:
                discount = np.tanh(steps*discount)
                # delta = reward + discount*max_next - Q[state_adj[0], state_adj[1], action]
                learning = np.tanh(beta*abs(delta)/steps)
            
            temp = c/steps
                
            # Adjust Q value for current state
            else:
                delta = learning*(reward + 
                                 discount*np.max(Q[state2_adj[0], 
                                                   state2_adj[1]]) - 
                                 Q[state_adj[0], state_adj[1],action])
                Q[state_adj[0], state_adj[1],action] += delta
                                     
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
        
        tot_episodes += 1

        # Track mean length of episodes and mean cumulative reward
        learn_len = (learn_len*(tot_episodes - 1) + steps)/tot_episodes
        learn_cumr = (learn_cumr*(tot_episodes - 1) + tot_reward)/tot_episodes

    test_len = 0
    test_cumr = 0
    tot_episodes = 0
    reached = 0
    # Run Adaptive Q learning algorithm: Testing Phase
    while reached < 40
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        pos, vel = test_st[reached]
        state = env.reset(pos, vel)
        steps = 0
        M = list()
        
        # Discretize state
        state_adj = discretize(state)
        state_tup = (state_adj[0], state_adj[1])
    
        while done != True:   
            # Determine next action - Boltzmann policy
            action = boltz_policy(state_adj[0], state_adj[1], Q, temp)
                
            # Get next state and reward
            state2, reward, done, info = env.step(action)
            steps += 1 
            
            # Discretize state2
            state2_adj = discretize(state2)
            state2_tup = (state2_adj[0], state2_adj[1])

            # Store s, a, r, s' in M
            if back:
                M.append((state_tup, action, reward, state2_tup))

            if done:
                reached += 1

            max_next = max(Q[state2_adj[0], state2_adj[1]])

            if ((th_n - k*steps) < max_next) and ((th_p + k*steps) > max_next) :
                discount = discount_init
                learning = learning_init
            else:
                discount = np.tanh(steps*discount)
                # delta = reward + discount*max_next - Q[state_adj[0], state_adj[1], action]
                learning = np.tanh(beta*abs(delta)/steps)
            
            temp = c/steps
                
            # Adjust Q value for current state
            else:
                delta = learning*(reward + 
                                 discount*np.max(Q[state2_adj[0], 
                                                   state2_adj[1]]) - 
                                 Q[state_adj[0], state_adj[1],action])
                Q[state_adj[0], state_adj[1],action] += delta
                                     
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
        
        tot_episodes += 1

        # Track mean length of episodes and mean cumulative reward
        test_len = (test_len*(tot_episodes - 1) + steps)/tot_episodes
        test_cumr = (test_cumr*(tot_episodes - 1) + tot_reward)/tot_episodes
            
    env.close()
    
    return learn_len, learn_cumr, test_len, test_cumr