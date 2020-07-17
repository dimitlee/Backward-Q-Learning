import numpy as np

from agents.policy import boltz_policy
from agents.discretizer import discretize
from const import TEMP, TEMP_MIN, VLOW, LOW, HIGH

def eql(env, learn_st, test_st, alpha=0.9, discount=0.99, back=False):
    
    # Initialize Q table
    Q = np.random.uniform(low = -1, high = 1, 
                          size = (11, 11, 
                                  env.action_space.n))

    learning_b = 0.05
    discount_b = 0.99
    visited = dict()
    temp = pow(10, HIGH)
    exp_weight = 0.7
    a_c = 10
    a_max = alpha

    # thresholds for q~ and exp
    q_thresh = 0.1
    exp_thresh = -4.5
    
    # Initialize variables to track rewards
    learn_len = 0
    learn_cumr = 0
    tot_episodes = 0
    
    reached = 0
    # Run Enhanced Q learning algorithm: Learning phase
    while reached < 300:
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        steps = 0
        x = np.random.choice(300)
        pos, vel = learn_st[x]
        state = env.reset(pos, vel)
        exp = 0.0
        M = list()
        
        # Discretize state
        state_adj = discretize(state)
        state_tup = (state_adj[0], state_adj[1])
    
        while done != True:   
            # Keep track of visiting number
            if state_tup in visited.keys():
                visited[state_tup] += 1
            else:
                visited[state_tup] = 1

            if steps == 0:
                state_prev = state_adj
            
            # compute q~, deltaV and exploration degree
            q_th = max(Q[state_adj[0], state_adj[1]]) - min(Q[state_adj[0], state_adj[1]])
            delta_v = max(Q[state_adj[0], state_adj[1]]) - max(Q[state_prev[0], state_prev[1]])
            exp = exp_weight*exp + (1 - exp_weight)*math.log(temp, 10)

            # compute temp using fuzzy balancer
            if (q_th < q_thresh and delta_v < 0):
                temp = pow(10, LOW)
            if (q_th < q_thresh and delta_v > 0 and exp < exp_thresh):
                temp = pow(10, VLOW)
            if (q_th < q_thresh and delta_v > 0 and exp > exp_thresh):
                temp = pow(10, LOW)
            if (q_th > q_thresh and delta_v < 0):
                temp = pow(10, LOW)
            if (q_th > q_thresh and delta_v > 0):
                temp = pow(10, LOW)
            
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

            # Compute adaptive learning rate
            alpha = min(a_c/visited[state_tup], a_max)

            #Allow for terminal states
            if done and state2[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward
                
            # Adjust Q value for current state
            else:
                delta = alpha*(reward + 
                                 discount*np.max(Q[state2_adj[0], 
                                                   state2_adj[1]]) - 
                                 Q[state_adj[0], state_adj[1],action])
                Q[state_adj[0], state_adj[1], action] += delta

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
    # Run Enhanced Q learning algorithm: Learning phase
    while reached < 300:
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        steps = 0
        pos, vel = test_st[reached]
        state = env.reset(pos, vel)
        exp = 0.0
        M = list()
        
        # Discretize state
        state_adj = discretize(state)
        state_tup = (state_adj[0], state_adj[1])
    
        while done != True:   
            # Keep track of visiting number
            if state_tup in visited.keys():
                visited[state_tup] += 1
            else:
                visited[state_tup] = 1

            if steps == 0:
                state_prev = state_adj
            
            # compute q~, deltaV and exploration degree
            q_th = max(Q[state_adj[0], state_adj[1]]) - min(Q[state_adj[0], state_adj[1]])
            delta_v = max(Q[state_adj[0], state_adj[1]]) - max(Q[state_prev[0], state_prev[1]])
            exp = exp_weight*exp + (1 - exp_weight)*math.log(temp, 10)

            # compute temp using fuzzy balancer
            if (q_th < q_thresh and delta_v < 0):
                temp = pow(10, LOW)
            if (q_th < q_thresh and delta_v > 0 and exp < exp_thresh):
                temp = pow(10, VLOW)
            if (q_th < q_thresh and delta_v > 0 and exp > exp_thresh):
                temp = pow(10, LOW)
            if (q_th > q_thresh and delta_v < 0):
                temp = pow(10, LOW)
            if (q_th > q_thresh and delta_v > 0):
                temp = pow(10, LOW)
            
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

            # Compute adaptive learning rate
            alpha = min(a_c/visited[state_tup], a_max)
                
            # Adjust Q value for current state
            else:
                delta = alpha*(reward + 
                                 discount*np.max(Q[state2_adj[0], 
                                                   state2_adj[1]]) - 
                                 Q[state_adj[0], state_adj[1],action])
                Q[state_adj[0], state_adj[1], action] += delta

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