"""
Name: Timothy McMahon
Student number: u5530441
"""

import numpy as np


def gen_matrix(length, b):
    """
    Generate invertible, quasipositive matrix
        """
    count = 0
    while True:
        mat = []
        for j in range(length):
            row = [0]*length
            i = 0
            while i < b:
                index = np.random.randint(length)
                if row[index] == 0:
                    row[index] = np.random.randint(0, 100)/100
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
            #check quasipositive:
                if False not in np.isreal(np.linalg.eig(mat)[0]):
                    return count
        count += 1
        if (count % 1000) == 0:
            print(count)
    # So what we do here 
    


    return count


def random_mdp(length, aggregation, num_actions, noise, b):
    """
    Generate a random MDP that is able to be v_aggregated
    """
    
    v = []
    
    for i in range(length):
    # Generate vector v of length s_phi from uniform distribution (0,100)
    # repeat the entries according to the aggregation factor, and add a small amount of random noise to each entry
        num = np.random.randint(100)
        for j in range(aggregation):
            # Add in random noise for each of these
            v.append(num + np.random.randint (-noise, noise)

    v = np.array(v)
    
    
    # Generate random Transition matrices

    transition_matrices = []

    for a in range(num_actions):
        flag = False
        while flag = False:
            mat = []
            for j in range(length*aggregation):
                row = [0]*length*aggregation
                i = 0
                while i < b:
                    index = np.random.randint(length*aggregation)
                    if row[index] == 0:
                        row[index] = np.random.rand()
                        i += 1

                # Normalise row so that the sum of the entries is 1
                row = np.array(row)
                total = row.sum()
                for i in range(len(row)):
                    row[i] = row[i]/total
                
                mat.append(row)
            
            # Check that the matrix is invertible
            if np.linalg.cond(mat) < 1/sys.float_info.epsilon:
                flag = True

        # So what we do here 
        mat = np.array(mat)
        
        transition_matrices.append(mat)
            

    # entries from uniform U(0,1), max of b elements (b is the branching factor)
    # normalise each row so that the sum of them is 1
    
    # then check if the matrix T is aperdiodic and irreducible.  
    
    
    # Solve for the optimal reward matrix/vector r
    # r = (T^-1 - \gamma I)v*
    
    # for each other action a-1
    # create q vectors q^a that are all  elementwise less than r
    # so just sample from U(0, r[i]) to create the vector., and add some random noise
    # and then r^a = T^-1(q^a - \gamma T v*)
    
    # When doing this entire thing, ensure that the policy is uniform.  I.e. that the optimal action 
    # for each of the states that will be aggregated is the same.  (
    # This should be fine by default. If the optimal action is always the same, then we are fine
   
    
    
    
    
    
    return 1
