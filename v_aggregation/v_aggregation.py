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
P = np.zeros((length, length))
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
            P[i][k] = temp
            
           
# Construct the reduced MDP



# Run VI over the MDP


# Run policy aggregation
