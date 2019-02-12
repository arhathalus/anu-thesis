
"""
Name: Timothy McMahon
Student number: u5530441
"""

from grid_world import NavGrid, QNavGrid, NavGridReduced, QNavGridReduced, QNavGridReducedLower
from agents import QLearn
import matplotlib.pyplot as plt

#TODO Write an abstraction function that takes in states and actions, and their transitions and values, and 
# produces an abstracted Q-value state (or value state)


def value_iteration(env, gamma, epsilon):
    """ Performs value iteration on an environment, given the parameters gamma and epsilon
        Returns a deterministic policy, and all the values for the states
        (navgrid, float, float) --> (dict, dict)
    """

    pi = {}

    states = []
    for i in range(env.num_states):
        state = env.state_to_pos(i)
        pi[state] = 'right'
        if state not in env.walls:
            states.append(state)

    # Initialise the values and policy
    values = {}            
    for state in states:
        values[state] = 0
    
    while True:
        delta = 0    
        for state in states:
            v = values[state]
            
            #calculate the max value functions
            val = -500
            for a in env.actions:
                # get the probabilites and the transitions
                temp_val = 0
                for i,p in enumerate(env.trans_probs[state, a]):
                    s_prime, r = env.generateReward(state, env.actions[i])
                    temp_val += p*(r + gamma*values[s_prime])
                if temp_val > val:
                    val = temp_val
                    pi[state] = a
            
            values[state] = val
            delta = max(delta, abs(v - values[state]))
            
        if delta < epsilon:
            break

    return pi, values


def value_iteration2(env, gamma, epsilon):
    """ Performs value iteration on an environment that requires gamma when
        generating reward, given the parameters gamma and epsilon
        Returns a deterministic policy, and all the values for the states
        (navgrid, float, float) --> (dict, dict)
    """

    pi = {}

    states = []
    for i in env.upper_indices:
        states.append(i)

    # Initialise the values
    values = {}            
    for state in states:
        values[state] = 0

    while True:
        delta = 0    
        for state in states:
            v = values[state]
            
            #calculate the max value functions
            val = -500
            for a in env.actions:
                # get the probabilites and the transitions
                temp_val = 0
                for i,p in enumerate(env.trans_probs[state, a]):
                    s_prime, r = env.generateReward(state, env.actions[i], gamma)
                    temp_val += p*(r + gamma*values[s_prime])
                if temp_val > val:
                    val = temp_val
                    pi[state] = a
            
            values[state] = val
            delta = max(delta, abs(v - values[state]))
            
        if delta < epsilon:
            break

    return pi, values

def lift_policy(policy, env):
    """
    Arguments: env: the environment to lift the policy to
               policy: the reduced policy
    Take the reduced policy and lift it up to the original environment
    Should be easy to do:
    We check what state we are in in the environment.
    If we are in the upper indices, we follow the policy
    If we are in the lower indices, we map the policy to the other direction
    And if we are on the diagonals, we go with 50/50 
    """
    
    new_policy = {}
    
    # Loop through all the states
    for i in range(env.num_states):
        # For every state, use the old policy to determine what is the direction
        state = env.state_to_pos(i)
            # on the diagonal, so doesn't matter which we choose
        if state[0] == state[1]:
            new_policy[state] = policy[state]
        # upper diagonal, so use the policy
        elif state[0] > state[1]:
            new_policy[state] = policy[state]
        # lower diagonals, so 
        else:
            # Find the appropriate upper index to sample in the old policy
            equiv_state = (state[1], state[0])
            if policy[equiv_state] == 'right':
                new_policy[state] = 'down'
            elif policy[equiv_state] == 'down':
                new_policy[state] = 'right'
            elif policy[equiv_state] == 'up':
                new_policy[state] = 'left'
            elif policy[equiv_state] == 'left':
                new_policy[state] = 'up'
    
    return new_policy


if __name__ == '__main__':
    
    # Initialise parameters
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.25
        
    # Make environment and learn optimal values using value iteration
    env = NavGrid(6,10,2)
    #env = NavGrid(10,15,10)
    pi, values = value_iteration(env, gamma, 0.00001)
    
    # Use these values to calculate the actual optimal policy

    #print(pi, values)
    env.print_map()
    print("-------")
    # Create modified grid world, and reduce it
    # The states of QNavGrid have the same values as NavGrid did
    env_q = QNavGrid(env, values)
    env_q_reduced = QNavGridReduced(env_q)
    env_q_reduced.print_map()

    # Do the same over the lower state reduction
    #env_q_reduced_lower = QNavGridReducedLower(env_q)
    #env_q_reduced_lower.print_map()

    
    # Now q-learn over the reduced environment and find optimal policy
    
    # Initialise the agent
    agent = QLearn(alpha, gamma, epsilon, env_q_reduced.num_states, env_q_reduced.actions)
    episode_length = []
    average_score = []

    num_episodes = 5000
    total_reward = 0
    ep_length = 0
    
    # Do the Q-learning in the reduced environment
    for episode in range(num_episodes):
        #print(agent.Q)

        action = agent.select_action(env_q_reduced.init_state)
        state = env_q_reduced.init_state

        while not env_q_reduced.end_of_episode():

            new_state, reward = env_q_reduced.generate(action, gamma)
            total_reward += reward
            ep_length += 1

            new_action = agent.select_action(new_state)
            agent.update_Q(state, action, reward, new_state)
            state = new_state
            action = new_action

    #for i in agent.Q:
    #    print(i, agent.Q[i])

    average_score.append(total_reward/num_episodes)
    episode_length.append(ep_length/num_episodes)
    
    # Take the q_values from the agent and determine optimal policy.
    policy = {}
    
    for i in range(env_q_reduced.num_states):
        _max = -99
        _action = ''
        for action in env_q_reduced.actions:
            if agent.Q[(i, action)] >= _max:
                _max = agent.Q[(i, action)]
                _action = action
        policy[i] = _action

    #env_q_reduced.print_map()
    
    #for i in policy:
    #    print(i, policy[i])
    
    
    p, v = value_iteration2(env_q_reduced, gamma, 0.001)
    #p2, v2 = value_iteration2(env_q_reduced_lower, gamma, 0.001)
    
    #for i in policy: 
    #    print(i)
    #    print("Q-Learning: " + policy[i])
    #    print("VI        : " + p[env_q_reduced.state_to_pos(i)])
        
    new_policy = lift_policy(p, env)

    #for i in new_policy:
    #    if new_policy[i] != pi[i]:
    #        if i not in env_q.walls:
    #            print("Disagreement at " + str(i), new_policy[i], pi[i])
                
    env_q.print_map_policy(new_policy)
    print("==========================")
    env_q.print_map_policy(pi)

    print(values)
#TODO:  Visualise this somehow properly when writing it up
# Explicitly calculate the expected reward for the policy, show it is optimal


# The whole point of the rebasing, doubling thing is that the action switches around, but the Q-Value 
# of the actions remain the same.  So this is correct, we get the optimal path is the original optimal path in
# the unreduced version.

# Might be good to visualise this at some point.