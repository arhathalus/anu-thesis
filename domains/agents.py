"""
Name: Timothy McMahon
Student number: u5530441
"""
import random, abc

class RLAgent(object):
    """ Generic class for RLAgent """
    
    def __init__(self, alpha, gamma, epsilon, num_states, actions):
        """ Make a new RLAgent with the specified parameters.
            (RLAgent, float, float, float, int, int) -> None
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon    
        self.num_states = num_states
        self.actions = actions
        #Q is a dictionary mapping (state, action) pairs to Q values
        self.Q = dict([((s, a), 0.0) for s in range(num_states) for a in actions])
    
    @abc.abstractmethod
    def select_action(self, state):
        """ This returns an action based on the value of the Q-table,
            and the exploration constant (epsilon). Break ties uniformly at random.
              (RLAgent, int) -> str
        """
        
    @abc.abstractmethod
    def update_Q(self):
        """  Update the Q-value table.
            (RLAgent, ??) -> None
        """

    def print_epsilon(self):
        print(self.epsilon)

class QLearn(RLAgent):
    """ The QLEarn agent implements the Q-learning algorithm.
    """
    def select_action(self, state):
        """ This returns an action based on the value of the Q-table,
            and the exploration constant (epsilon). Break ties uniformly at random.

            (RLAgent, int) -> str
        """
        if random.random() <= self.epsilon:
            # choose element randomly
            return random.choice(self.actions)
        else:
            # Choose the element with the highest Q values
            max_actions = [x for x in self.actions if self.Q[state, x] == max([self.Q[state, k] for k in self.actions])]
            return random.choice(max_actions)

    def update_Q(self, state, action, reward, state1):
        """  Update the Q-value table. 
            (RLAgent, ??) -> None
        """
        self.Q[state, action] += self.alpha * (reward + self.gamma * max([self.Q[state1, a] for a in self.actions])
                                               - self.Q[state, action])