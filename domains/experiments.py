
"""
Name: Timothy McMahon
Student number: u5530441
"""

from environment import NavGrid, QNavGrid, NavGridReduced, QNavGridReduced
#from agents import Sarsa, QLearn
import matplotlib.pyplot as plt

def value_iteration(env, gamma, epsilon):
    """ Performs value iteration on an environment, given the parameters gamma and epsilon
        Returns a deterministic policy, and all the values for the states
        (navgrid, float, float) --> (dict, dict)
    """

    pi = {}

    states = []
    for i in range(env.num_states):
        state = env.state_to_pos(i)
        if state not in env.walls:
            states.append(state)

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



if __name__ == '__main__':
    
# TODO 
# make environment and agent
# learn the q-values using q-learning or VI or both and compare
# reduce under homomorphism, learn q values
# find deterministic policy, uplift to bigger domain, check performance

#learn the q-values using q-learning or VI or both and compare
# construct q-value constant reflection, but break MDPness of homomorphism
# reduce,
# learn,
# test performance