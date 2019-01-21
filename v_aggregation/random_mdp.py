"""
Name: Timothy McMahon
Student number: u5530441
"""

import numpy as np
import sys


def gen_matrix(length, b, epsilon):
    """
    Generates an invertible, quasipositive matrix of dimensions length x length
    length: int
    b: int - the branch factor, the number of non-negligable transition probabilities
    epsilon: double:  the small probability of transitioning to any state.  This should be close to 0.
        """
    mat = []
    for j in range(length):
        # Initialise the matrix with epsilon entries
        row = [epsilon]*length

        # add transition probabilites to b random entries
        i = 0
        while i < b:
            index = np.random.randint(length)
            if row[index] == epsilon:
                # Sample from the interval [epsilon, 1)
                row[index] = (1-epsilon)*np.random.random_sample() + epsilon
                i += 1

        # Normalise row so that the sum of the entries is 1
        row = np.array(row)
        total = row.sum()
        for i in range(len(row)):
            row[i] = row[i]/total
        
        mat.append(row)
    
    mat = np.array(mat)
    
    # Check that the matrix is invertible
    if np.linalg.cond(mat) < 1/sys.float_info.epsilon:
        return mat
    else:
        return 0


def random_mdp(length, aggregation, num_actions, noise, b, epsilon, gamma):
    """
    Generate a random MDP that is able to be v_aggregated
    """
    
    v = []
    
    for i in range(length):
    # Generate vector v of length s_phi from uniform distribution (0,100)
    # repeat the entries according to the aggregation factor, and add a small amount of random noise to each entry
        num = 100*np.random.random_sample()
        for j in range(aggregation):
            # Add in random noise for each of these
            v.append(num + 2*noise*np.random.random_sample() -noise)

    v = np.array(v)
    
    # Generate random transition matrices, that are invertible and quasipositive

    transition_matrices = []

    while len(transition_matrices) < num_actions:
            temp_mat = gen_matrix(length*aggregation, b, epsilon)
            if type(temp_mat) != int:
                transition_matrices.append(temp_mat)

   
    # Generate the q-value vectors
    q_vals = {}
    i = 0
    while len(q_vals) < num_actions:
        
        noise_vec = 2*noise*np.random.random_sample(length*aggregation) - noise
        q_vec = np.array(np.random.random_sample(length*aggregation))*v + noise_vec
        # Double check that everything is less than the v vector
        if False not in (q_vec < v):
            q_vals[i] = q_vec
            i += 1
          
    rewards = []
    transition_inverse = np.linalg.inv(transition_matrices[0])
    # Solve for the optimal reward matrix/vector r
    # r = (T^-1 - \gamma I)v*
    r = np.matmul(transition_inverse - gamma*np.identity(length*aggregation), v)
    rewards.append(r)
    
    # find the rest of the reward vectors
    # r_a = T^-1(q_a - \gamma T v*)
    for i in range(1, num_actions):
        #transition_inverse = np.linalg.inv(transition_matrices[i])
        r = np.matmul(transition_inverse, q_vals[i] - np.matmul(gamma*transition_matrices[0], v))
        rewards.append(r)

    #TODO Confirm that i can just use the one transition matrix here.
    
    # When doing this entire thing, ensure that the policy is uniform.  I.e. that the optimal action 
    # for each of the states that will be aggregated is the same.  (
    # This should be fine by default. If the optimal action is always the same, then we are fine
   
    return v, transition_matrices, q_vals, rewards

length = 2
aggregation = 2
num_actions = 2
noise = 1
b = 2
epsilon = 0.0005
gamma = 0.8
eps = 0.001

v, transition_matrices, q_vals, rewards = random_mdp(length, aggregation, num_actions, noise, b, epsilon, gamma)
#print(random_mdp(length, aggregation, num_actions, noise, b, epsilon, gamma))

print("values:")
print(v)
#print("T:")
#print(transition_matrices)
#print("Q:")
#print(q_vals)
#print("R:")
#print(rewards)

 # Initialise the values
values = [0]*(length*aggregation)            
eps = 0.000000000001
pi = {}

while True:
    delta = 0    
    for state in range(length*aggregation):
        temp_v = values[state]
        
        #calculate the max value functions
        val = -500
        for a in range(num_actions):
            # get the probabilites and the transitions
            temp_val = 0
            for s_prime in range(length*aggregation):
                r = rewards[a][s_prime]
                temp_val += transition_matrices[a][state][s_prime]*(r + gamma*values[s_prime])

            if temp_val > val:
                val = temp_val
                pi[state] = a
        
        values[state] = val
        delta = max(delta, abs(temp_v - values[state]))
  
    if delta < eps:
        break



print("learned values ")
print(values)
#print(v)
print("Policy")
print(pi)    

#from scipy.linalg import eig
#for a in range(num_actions):
#
#    # solve the stationary distribution p_a T_a = p_a (the left eigenvector)
#    e, vl = eig(transition_matrices[a], left=True, right=False)
#    
#    # Pull out the eigenvalue that is equal to 1
#    index = np.where(np.isclose(np.real(e), 1))
#    print(index)
#    # create the left eigenvector (and cast to real as well)
#    print(vl[:,index[0][0]].T)
#    print(np.real(vl[:,index[0][0]].T))


# Two issues
# Im sometimes getting optimal action is 1, but it should be 0, and VI learns the wrong values
# Need to charactrise the problem here --- what do the bad domains look like?
# Sometimes it is working, other times VI doesnt find it at all and i am getting wrong optimal actions
# Question: is something going wrong when i learn the wrong thing?  or is it just a VI problem somehow?



# left eigenvectors are returning as constant. (Why is this?  check this out)
# I have fixed some eigenvector problems, but check this later to make sure i have no more problems.