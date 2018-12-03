"""
Name: Timothy McMahon
Student number: u5530441
"""

import numpy as np
from scipy.linalg import eig

from random_mdp import gen_matrix, random_mdp

length = 4
aggregation = 2
num_actions = 2
noise = 0
b = 3
epsilon = 0.05
gamma = 0.4

v, transition_matrices, q_vals, rewards = random_mdp(length, aggregation, num_actions, noise, b, epsilon, gamma)

# Reduce the MDP
B = {}
# initialise the new transition matrices
P = []
for _ in range(num_actions):
    P.append(np.zeros((length, length)))
    
# Construct stochastic inverse
for a in range(num_actions):

    # solve the stationary distribution p_a T_a = p_a (the left eigenvector)
    e, vl = eig(transition_matrices[a])
    # Sanity check that the eigenvalue is 1
    assert(e[0] - 1 < 0.001)

    rho = vl[:,0].T
    
    # calculate the stochastic inverse 
    # B(s | phi(s), a)
    denoms = []
    for i in range(length):
        denom = 0
        for j in range(aggregation):
            denom += rho[i*aggregation+j]
        
        denoms.append(denom)
    
    for i in range(length):
        for j in range(aggregation):
        #B[s, phi_s, a] = rho(s)/denom[phi_s]
            B[(i*aggregation+j, i, a)] = rho[i*j+j]/denoms[i]
    
    
    # calculate the new prob distribution
    for i in range(length):
        for k in range(length):
            #p[i][k]
            temp = 0
            # calculate the prob here
            # get all the things that map to i
            # there are aggregation number of them
            for j in range(aggregation):
                for l in range(aggregation):
                  temp += B[i*aggregation+j, i, a]*transition_matrices[a][i*aggregation+j][k*aggregation+l]  
            P[a][i][k] = temp
            
           
# Run VI over the MDP to determine v* and optimal policy

# TODO Check this, i haven't made a new rewards function

# Initialise the values
values = [0]*length            
eps = 0.001
pi = {}

while True:
    delta = 0    
    for state in range(length):
        temp_v = values[state]
        
        #calculate the max value functions
        val = -500
        for a in range(num_actions):
            # get the probabilites and the transitions
            temp_val = 0
            for s_prime in range(length):
                r = rewards[a][s_prime]
                temp_val += P[a][state][s_prime]*(r + gamma*values[s_prime])

            if temp_val > val:
                val = temp_val
                pi[state] = a
        
        values[state] = val
        delta = max(delta, abs(temp_v - values[state]))
        
    if delta < eps:
        break


# Learn optimal lifted policy
lifted_policy = {}
for s in range(length*aggregation):
    lifted_policy[s] = pi[np.floor(s/aggregation)]
    
# Perform policy iteration
bigValues = [0]*length*aggregation

while True:
    delta = 0    
    for state in range(length*aggregation):
        temp_v = bigValues[state]
        
        # get action from policy:
        a = lifted_policy[state]

        # get the probabilites and the transitions
        temp_val = 0
        for s_prime in range(length*aggregation):
            r = rewards[a][s_prime]
            temp_val += transition_matrices[a][state][s_prime]*(r + gamma*bigValues[s_prime])

        
        bigValues[state] = temp_val
        delta = max(delta, abs(temp_v - bigValues[state]))
        
    if delta < eps:
        break


#TODO Ask Sultan -- if i perform the policy learning whilst doing value iteration, do i lose anything?

# Run policy aggregation
