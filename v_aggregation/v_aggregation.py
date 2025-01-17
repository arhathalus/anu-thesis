"""
Name: Timothy McMahon
Student number: u5530441
"""

import pickle
import numpy as np
from scipy.linalg import eig
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from agents import QLearn

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
        rho = np.abs(np.real(vl[:,index[0][0]].T))
        
        # normalise the eigenvector to form the stationary distribution
        new_rho = rho/np.sum(rho)
        #print(rho)
        rhos.append(new_rho)

        # for all states that map to s, sum the rho of that action
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
                B[(i*aggregation+j, i, a)] = rho[i*aggregation+j]/denoms[i]
        
        # calculate the new prob distribution and reward matrices
        for i in range(length):
            for k in range(length):
                temp_prob = 0
                temp_reward = 0

                # get all the things that map to i
                # there are aggregation number of them
                z = 0
                for j in range(aggregation):
                    for l in range(aggregation):
                        temp_prob += B[i*aggregation+j, i, a]*transition_matrices[a][i*aggregation+j][k*aggregation+l]  
                    temp_reward += B[i*aggregation+j, i, a]*rewards[a][i*aggregation+j]

                P[a][i][k] = np.real(temp_prob)
                R[a][i] = np.real(temp_reward)

    # calculate the stationary distribution for the aggregated MDP
    aggregated_rhos = []
    for a in range(num_actions):
        # solve the stationary distribution p_a T_a = p_a (the left eigenvector)
        e, vl = eig(P[a], left=True, right=False)

        # Pull out the eigenvalue that is equal to 1
        index = np.where(np.isclose(np.real(e), 1))

        # create the left eigenvector (and discard the imaginary part (which should be zero)
        temp_rho = np.abs(np.real(vl[:,index[0][0]].T))
        
        # normalise the eigenvector to form the stationary distribution
        new_rho = temp_rho/np.sum(temp_rho)
        #print(rho)
        aggregated_rhos.append(new_rho)


    # Run VI over the aggregated MDP to determine v* and optimal policy

    # Initialise the values
    values = [0]*length            

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
            
            values[state] = val
            delta = max(delta, abs(temp_v - values[state]))
            
        if delta < eps:
            break

    # Learn the real-optimal policy directly from the actual values v
    #real_pi = {}
    
    #for state in range(length*aggregation):
        #first_flag = True
        #temp_v = v[state]
            
        ##calculate the max value functions
        #for a in range(num_actions):
            ## get the probabilites and the transitions
            #temp_val = 0
            #for s_prime in range(length*aggregation):
                #r = rewards[a][s_prime]
                #temp_val += transition_matrices[a][state][s_prime]*(r + gamma*v[s_prime])
            #if first_flag:
                #first_flag = False
                #real_pi[state] = a
                #val = temp_val
            #elif temp_val > val:
                #val = temp_val
                #real_pi[state] = a

    # Learn the optimal policy
    pi = {}
    
    for state in range(length):
        first_flag = True
        temp_v = values[state]
            
        #calculate the max value functions
        for a in range(num_actions):
            # get the probabilites and the transitions
            temp_val = 0
            for s_prime in range(length):
                r = R[a][s_prime]
                temp_val += P[a][state][s_prime]*(r + gamma*values[s_prime])
            if first_flag:
                first_flag = False
                pi[state] = a
                val = temp_val
            elif temp_val > val:
                val = temp_val
                pi[state] = a
        
    
    # Q-Learn over the aggregated MDP to v* and optimal policy
    #agent = QLearn(0.4, gamma, epsilon, length, [0,1])
    
    #action = agent.select_action(0)
    #state = 0
    #ep_length = 0
    #while ep_length < 8000:

        #new_state = np.random.choice(range(length), 1, p=P[action][state])[0]
        #reward = R[action][new_state]
        #ep_length += 1

        #new_action = agent.select_action(new_state)
        #agent.update_Q(state, action, reward, new_state)
        #state = new_state
        #action = new_action

   
    ## Take the q_values from the agent and determine optimal policy.
    #q_policy = {}
    
    #for i in range(length):
        #_max = -99999
        #_action = 0
        #for action in range(num_actions):
            #if agent.Q[(i, action)] >= _max:
                #_max = agent.Q[(i, action)]
                #_action = action
        #q_policy[i] = _action
 
 
    #print(q_policy)
    #print(pi)

    
    #pi=q_policy
    
    
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
        #rho_vec.append(temp)

    MDP_measure_vec = []

    for s in range(aggregation):
        a = pi[s]
        temp_val = 0
        for j in range(aggregation):
            temp_val += B[aggregation*s+j, s, a] * np.abs((bigValues[aggregation*s+j] - v[aggregation*s+j]))
    
        MDP_measure_vec.append(aggregated_rhos[a][s]*temp_val)


    # Return learned optimal policy, optimal values , learned values, learned aggregate policy
    return v, bigValues, values, lifted_policy, pi, rho_vec, B, aggregated_rhos, sum(MDP_measure_vec)


#noises = [1,5,10,15,20]
#aggregations = [2,4,16,32]
length = 8
aggregation = 4
num_actions = 2
b = 4
#epsilon = 0.0005
epsilon = 0.0000000005
gamma = 0.8
eps = 0.00000001
noise = 1 


#for noise in noise: 
#    v, bigValues, values, lifted_policy, pi, rho_vec = experiment(length, aggregation, num_actions, noise, b, epsilon, gamma, eps)

#for i in range(100):
#noise = 5
    #v, bigValues, values, lifted_policy, pi, rho_vec = experiment(length, aggregation, num_actions, noise, b, epsilon, gamma, eps)    
    
# Want to graph np.abs(v - bigValues) against rho_vec

#print(pi)
#print("--------------------")
#print(lifted_policy)

#print(np.abs(v-bigValues))
#print(rho_vec)

# For each parameter set, generate 1000 MDPs, calculate the two different vectors
# and calculate the pearson correlation coefficent for them all.
# See if we get similar results


all_rho_vec = []
abs_values = []

v_vec = []
bigValues_vec = []
lifted_policy_vec = []
values_vec = []
pi_vec = []
B_vec = []
rho_stored_vec = []
stored_aggregated_rhos = []
mdp_measures = []
good_rho_vec = []
good_abs_values = []

# v are the original values
# bigValues are the learned values for the original MDP
# lifted policy : the learned policy for the original MDP
# values are the learned values in the aggregate MDP
# pi is the learned policy in the aggregate MDP
# B the prob function
# rho_vec is the vector 1/np.real(rhos[lifted_policy[i]][i])

i = 0
j = 0
while i < 1000:
    if i%100 == 0:
        print(i)
    v, bigValues, values, lifted_policy, pi, rho_vec, B, aggregated_rhos, mdp_measure = experiment(
        length, aggregation, num_actions, noise, b, epsilon, gamma, eps)     
    j += 1
    if 1 in lifted_policy.values():
        all_rho_vec += rho_vec
        abs_values += np.abs(v-bigValues).tolist()
        #Extra stuff to save all the results
        v_vec.append(v)
        bigValues_vec.append(bigValues)
        values_vec.append(values)
        lifted_policy_vec.append(lifted_policy)
        pi_vec.append(pi)
        B_vec.append(B)
        rho_stored_vec.append(rho_vec)
        stored_aggregated_rhos.append(aggregated_rhos)
        mdp_measures.append(mdp_measure)
        i += 1
    else:
        good_rho_vec += rho_vec
        good_abs_values += np.abs(v-bigValues).tolist()
        

print(j)

#pickle_data = [B_vec, v_vec, bigValues_vec, lifted_policy_vec, values_vec, pi_vec, rho_stored_vec, stored_aggregated_rhos, mdp_measures]
#with open('timData3.pkl', 'wb') as f:
#    pickle.dump(pickle_data, f)
#print(all_rho_vec)
#print(abs_values)
#Extra stuff to save all the results
#print(v_vec)
#print(bigValues_vec)
#print(values_vec)
#print(lifted_policy_vec)
#print(pi_vec)

#print(j)
#print(noise)
#print(length)
#print(aggregation)
#print(np.corrcoef(all_rho_vec, abs_values))
#print(pearsonr(all_rho_vec, abs_values))

fix, ax = plt.subplots()
ax.scatter(range(len(mdp_measures)), mdp_measures)
#plt.xlabel(r'$\log_2 \frac{1}{\rho^{\tilde{\Pi}(s)} }$')
#plt.ylabel(r'$| V^*(s) - V^{ \tilde{\Pi} }(s) |$')
#plt.title('Difference between the true and learned values vs the inverse stationary distribution')
plt.show()


fix, ax = plt.subplots()
ax.scatter(good_rho_vec, good_abs_values, s=.25)
plt.xlabel(r'$\log_2 \frac{1}{\rho^{\tilde{\Pi}(s)} }$')
plt.ylabel(r'$| V^*(s) - V^{ \tilde{\Pi} }(s) |$')
#plt.title('Difference between the true and learned values vs the inverse stationary distribution')
plt.show()


fix, ax = plt.subplots()
ax.scatter(all_rho_vec, abs_values, s=.25)
plt.xlabel(r'$\log_2 \frac{1}{\rho^{\tilde{\Pi}(s)} }$')
plt.ylabel(r'$| V^*(s) - V^{ \tilde{\Pi} }(s) |$')
#plt.title('Difference between the true and learned values vs the inverse stationary distribution')
plt.show()




# v is the real values of the mdp
# bigValues are the learned values
# values are the learned values in the reduced_mdp
#TODO These are very large : check if this always happens : and if that means there is a problem in the code

# I am occassionally learning the wrong action here.  Sometimes for all states, sometimes not.
# This is expected --- V aggregation doesn't always converge optimally.  


# V* is v
# V^Pi is bigValues
# i think this is what i want to calculate

