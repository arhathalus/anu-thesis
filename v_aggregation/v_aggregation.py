"""
Name: Timothy McMahon
Student number: u5530441
"""

import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

from random_mdp import gen_matrix, random_mdp

def experiment(length, aggregation, num_actions, noise, b, epsilon, gamma, eps):

    v, transition_matrices, q_vals, rewards = random_mdp(length, aggregation, num_actions, noise, b, epsilon, gamma)

    # Reduce the MDP
    B = {}
    # initialise the new transition matrices
    P = []
    R = []
    for _ in range(num_actions):
        P.append(np.zeros((length, length)))
        R.append(np.zeros((length)))
    rhos = []    
    # Construct stochastic inverse
    for a in range(num_actions):

        # solve the stationary distribution p_a T_a = p_a (the left eigenvector)
        e, vl = eig(transition_matrices[a], left=True, right=False)

        # Pull out the eigenvalue that is equal to 1
        index = np.where(np.isclose(np.real(e), 1))

        # create the left eigenvector (and discard the imaginary part (which should be zero)
        rho = np.real(vl[:,index[0][0]].T)
        rhos.append(rho)


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
        
        
        # calculate the new prob distribution and reward matrices
        for i in range(length):
            for k in range(length):
                #p[i][k]
                temp_prob = 0
                temp_reward = 0
                # calculate the prob here
                # get all the things that map to i
                # there are aggregation number of them
                z = 0
                for j in range(aggregation):
                    for l in range(aggregation):
                        temp_prob += B[i*aggregation+j, i, a]*transition_matrices[a][i*aggregation+j][k*aggregation+l]  
                    temp_reward += B[i*aggregation+j, i, a]*rewards[a][i*aggregation+j]

                P[a][i][k] = np.real(temp_prob)
                R[a][i] = np.real(temp_reward)
                
            
    # Run VI over the MDP to determine v* and optimal policy

    # Initialise the values
    values = [0]*length            
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
                    r = R[a][s_prime]
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
        lifted_policy[s] = pi[int(np.floor(s/aggregation))]
        
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

    # Calculate the vector of log(1/p^PI(s))
    rho_vec = []
    for i in range(length*aggregation):
        # check what action is taken in state s
        temp =  1/np.real(rhos[lifted_policy[i]][i])
        rho_vec.append(np.log2(temp))


    return v, bigValues, values, lifted_policy, pi, rho_vec


        
    
#test_vec = []
#for i in lifted_policy.keys():
#    test_vec.append(np.log2(1/rhos[lifted_policy[i]][i]))
#print(test_vec)

#TODO Ask Sultan -- if i perform the policy learning whilst doing value iteration, do i lose anything?


# Need to wrap this entire thing in a method, and then do it for the different values used in Boris' Thesis.
# Want to calculate |V* - V^PI| and compare to 1/p^PI


# This should be done with noise values of 1,5,10,15,20
# and aggregation factors 2,4,16,32
# graph results , first lets create graph similar to the one in the paper . 
#2 actions, 64 states, aggregation factor 16, branching factor 4, noise 5

noises = [1,5,10,15,20]
aggregations = [2,4,16,32]
length = 2
aggregation = 2
num_actions = 2
b = 2
epsilon = 0.0005
gamma = 0.8
eps = 0.000001


#for noise in noise: 
#    v, bigValues, values, lifted_policy, pi, rho_vec = experiment(length, aggregation, num_actions, noise, b, epsilon, gamma, eps)
noise = 5
v, bigValues, values, lifted_policy, pi, rho_vec = experiment(length, aggregation, num_actions, noise, b, epsilon, gamma, eps)    
    
# Want to graph np.abs(v - bigValues) against rho_vec

print(pi)
print("--------------------")
print(lifted_policy)

#print(np.abs(v-bigValues))
#print(rho_vec)


#fix, ax = plt.subplots()
#ax.scatter(rho_vec, np.abs(v-bigValues))
#plt.show()


# I am occassionally learning the wrong action here.  Sometimes for all states, sometimes not.

# I am getting negative eigenvectors somehow here as well, which is causing problems.