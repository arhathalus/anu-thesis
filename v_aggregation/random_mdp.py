"""
Name: Timothy McMahon
Student number: u5530441
"""

import numpy as np


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


def random_mdp(length, aggregation, num_actions, noise, b, epsilon):
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

    while len(transition_matrices) < num_actions
            temp_mat = gen_matrix(length, b, epsilon)
            if temp_mat:
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
          
    
    # Solve for the optimal reward matrix/vector r
    # r = (T^-1 - \gamma I)v*
    r = np.matmul((np.linalg.inv(transition_matrices[0]) - gamma*np.identity(length*aggregation)), v)
    # and then r^a = T^-1(q^a - \gamma T v*)
    
    # When doing this entire thing, ensure that the policy is uniform.  I.e. that the optimal action 
    # for each of the states that will be aggregated is the same.  (
    # This should be fine by default. If the optimal action is always the same, then we are fine
   
    
    
    
    
    
    return 1
