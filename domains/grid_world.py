# Skeleton of code taken from COMP6320 Artificial Intelligence
# ANU

""" This file implements the domains NavGrid and QNavGrid, and reduced versions of them
"""

import random, abc
import numpy as np
import itertools
import copy


def adjcells(grid, state):
    """ Take a grid and state, and return true if placing a 
        wall in that state won't make the grid unsolvable
        (dict, (x,y) -> bool)
    """
    x,y = state[0], state[1]
    colours = set([])
    for i in range(-1,2):
        for j in range(-1,2):
            if grid[x+i, y+j] == 'B':
                colours.add('B')
            elif grid[x+i, y+j] == 'R':
                colours.add('R')
    if len(colours) == 0:
        grid[x,y] = '#'
        return True
    elif len(colours) == 1:
        c = colours.pop()
        grid[x,y] = c
        grid[x,y] = c
        for i in range(-1, 2):
            for j in range(-1,2):
                if grid[x+i, y+j] == '#':
                    adjcells(grid, (x+i, y+j))
        return True
    else:
        return False


class Environment(object):
    """ The class Environment will be inherited by all of the types of
        environment in which we will run experiments.
    """
    
    @abc.abstractmethod
    def __init__(self, width, height, init_pos, goal_pos):
        """ All Environments share these common features.
            
            (Environment, int, int, (int, int), (int, int)) -> None
        """
        self.width = width
        self.height = height
        self.init_pos = init_pos
        self.goal_pos = goal_pos
        
        self.num_states = width * height
        self.init_state = self.pos_to_state(init_pos)
        self.current_pos = init_pos
        
        self.num_actions = 4
        self.actions = ["up", "down", "left", "right"]
        self.action_dirs = {"up"    : (0, -1),
                            "down"  : (0, 1),
                            "left"  : (-1, 0),
                            "right" : (1, 0) }

    @abc.abstractmethod
    def generate(self, action):    
        """ Apply the given action to the current state and return
            an (observation, reward) pair.
            (Environment, str) -> ((int, int), float)
        """

    def end_of_episode(self):
        """ Return iff it is the end of the episode. If so, reset the
            environment to the initial state.
            (Environment) -> bool
        """
        if self.current_pos == self.goal_pos:   
            self.current_pos = self.init_pos
            self.init_state = self.pos_to_state(self.init_pos)  
            return True
        return False

    def pos_to_state(self, pos):
        """ Return the index (representing the state) of the current position.
            (Environment, (int, int)) -> int
        """
        return pos[1]*self.width + pos[0]

    def state_to_pos(self, state):
        """ Return the index (representing the state) of the current position.
            (Environment, (int, int)) -> int
        """
        return (state//self.width, state % self.width)
    

class NavGrid(Environment):
    """ NavGrid environment 
        The grid is symmetrical along the diagonal, except for a specified number of walls
    """

    def __init__(self, width, num_walls, num_antisym, walls_list=None, trans_probs=None):
        """ Make a new NavigationalGrid environment.
            width: int
            num_walls : the number of walls to be placed in the grid
            num_antisym : the number of walls not reflected on the diagonal
            walls_list : optional array that specifies the location of the walls in the grid
            trans_probs: optional dict that specifies the state transition probabilities
            (NavGrid, int, int, int, array, dict) --> int
        """

        super(NavGrid, self).__init__(width, width, (0,0), (width-1, width-1))
        
        # Use wall list if provided, else randomly generate the walls
        if walls_list is None:
            self.walls = set()
            self.anti_sym = set()
            grid = {}
            
            for i in range(width):
                for j in range(width):
                    grid[i,j] = '.'
            
            # Colour the edges of the grid 
            # will use later when determining if placing a wall will make the grid unsolvable
            for i in range(width+1):
                grid[-1, i] = 'R'
                grid[i, width] = 'R'
                grid[i, -1] = 'B'
                grid[width, i] = 'B'
            
            grid[-1,-1] = '#'
            grid[width, width] = '#'
            

            allcells = [x for x in itertools.product(range(width), range(width))] 
            random.shuffle(allcells)
            
            # Place the walls in the grid, ensuring the grid is always completable
            i = 0
            while(len(self.walls) < num_walls or len(self.walls) < num_antisym) :

                # add in cell and its reflection if it doesn't cut off a path
                if adjcells(grid, allcells[i]):
                    if len(self.anti_sym) >= num_antisym:
                        if adjcells(grid, (allcells[i][1], allcells[i][0])):
                            self.walls.add(allcells[i])
                            self.walls.add((allcells[i][1], allcells[i][0]))
                        else:
                            grid[allcells[i]] = '.'
                    else:
                        self.walls.add(allcells[i])
                        self.anti_sym.add(allcells[i])
       
                i += 1
                if i == len(allcells):
                    break

        # Use the wall list if provided        
        else:
            self.walls = walls_list
        
        ## Debug code to print the grid output
        #for i in range(-1, width+1):
        #    for j in range(-1, width+1):
        #        print(grid[i,j], end="")
        #    print("")
            
        # Use trans_probs if provided, else set the transition probabilities to the default values
        if trans_probs is None:
            coords = list(itertools.product(range(width), range(width)))
            self.trans_probs = {}
            for coord in coords:
                for i in range(len(self.actions)):
                    self.trans_probs[(coord, self.actions[i])] = [0, 0, 0, 0]
                    self.trans_probs[(coord, self.actions[i])][i] = 1
        else:
            self.trans_probs = trans_probs
    
    
    def generate(self, action):
        """ Apply the given action to the current state and return
            an (observation, reward) pair.
            (NavGrid, str) -> (int, float)
        """
        # Get the direction we go given our action
        a_action = np.random.choice(self.actions, 1, p=self.trans_probs[self.current_pos, action])[0]
        a_dir = self.action_dirs[a_action]
        
        #Clever min-max bounds checking
        pos = self.current_pos
        pos = (min(max(pos[0] + a_dir[0], 0), self.width-1),
               min(max(pos[1] + a_dir[1], 0), self.height-1))
        
        # If we haven't moved into a wall, update the position
        if pos not in self.walls:
            self.current_pos = pos
        
        if self.current_pos == self.goal_pos:
            r = 10.0
        else:
            r = -1.0

        return (self.pos_to_state(self.current_pos), r)
    
    
    def generateReward(self, state, action):
        """ Apply the given action to the given state and return
            an (observation, reward) pair.  This doesn't update the agent's position
            (NavGrid, (int, int), str) -> (int, float)
        """
        # Get the direction we go given our action
        a_action = np.random.choice(self.actions, 1, p=self.trans_probs[state, action])[0]
        a_dir = self.action_dirs[a_action]
        
        #Clever min-max bounds checking
        pos = state
        pos = (min(max(pos[0] + a_dir[0], 0), self.width-1),
               min(max(pos[1] + a_dir[1], 0), self.height-1))
        
        # If we haven't moved into a wall, update the position
        if pos in self.walls:
            pos = state
        
        if pos == self.goal_pos:
            r = 10.0
        else:
            r = -1.0
        return pos, r
    
   
    def print_map(self):
        """ Print an ASCII map of the simulation.
            (NavGrid) -> None
        """
        for y in range(self.height):
            for x in range(self.width):
                pos = (x, y)
                if pos == self.current_pos:
                    print("A ", end=''),
                elif pos == self.init_pos:
                    print("S ", end=''),
                elif pos == self.goal_pos:
                    print("G ", end=''),
                elif pos in self.walls:
                    print("X ", end=''),
                else:
                    print("* ", end=''),
            print("") 
    

class NavGridReduced(NavGrid):
    """ Takes a NavGrid class and reduces it under a MDP homomorphism.  
        This reduces the grid to the upper diagonal.
    """

    def __init__(self, navgrid):
        """ Make a new NavGridReduced environment.
            (NavGridReduced, NavGrid) -> int
        """
        # Get the upper indices  of the original grid
        self.upper_indices = [(i,j) for i in range(navgrid.width) for j in range(navgrid.width) if i >= j]
        super(NavGridReduced, self).__init__(navgrid.width, len(navgrid.walls), len(navgrid.anti_sym), navgrid.walls, navgrid.trans_probs)
                
 
    def generate(self, action):
        """ Apply the given action to the current state and return
            an (observation, reward) pair.
            (NavGridReduced, str) -> (int, float)
        """
        # Get the direction we go given our action
        a_action = np.random.choice(self.actions, 1, p=self.trans_probs[self.current_pos, action])[0]
        a_dir = self.action_dirs[a_action]
        
        #Clever min-max bounds checking
        pos = self.current_pos
        pos = (min(max(pos[0] + a_dir[0], 0), self.width-1),
               min(max(pos[1] + a_dir[1], 0), self.height-1))
 
        # If we haven't moved into a wall, update the position
        # (needs to be in the upper_indices as well)
        if pos not in self.walls and pos in self.upper_indices:
            self.current_pos = pos
        
        if  self.current_pos == self.goal_pos:
            r = 10.0
        else:
            r = -1.0
        return (self.pos_to_state(self.current_pos), r)


    def pos_to_state(self, pos):
        """ Return the index (representing the state) of the current position.
            #TODO: Check if this is fine
            (Environment, (int, int)) -> int
        """
        return self.upper_indices.index(pos)
    
    def state_to_pos(self, state):
        """ Return the index (representing the state) of the current position.
            (Environment, (int, int)) -> int
        """
        return (self.upper_indices[state])
    
    
    def print_map(self):
        """ Print an ASCII map of the simulation.
            (NavGrid) -> None
        """
        for y in range(self.height):
            for x in range(self.width):
                pos = (x, y)
                if pos in self.upper_indices:
                    if pos == self.current_pos:
                        print("A ", end=''),
                    elif pos == self.init_pos:
                        print("S ", end=''),
                    elif pos == self.goal_pos:
                        print("G ", end=''),
                    elif pos in self.walls:
                        print("X ", end=''),
                    else:
                        print("* ", end=''),
                else:
                    print("  ", end=''),
            print("") 


class QNavGrid(NavGrid):
    """ Takes a NavGrid as input, and produces a new navgrid that has approximate q-value symmetry
        along the diagonal.
    """

    def __init__(self, navgrid, values):
        """ Make a new NavigationalGrid environment.
            (QNavGrid, navgrid, dict) --> int
        """

        super(QNavGrid, self).__init__(navgrid.width, len(navgrid.walls), len(navgrid.anti_sym), navgrid.walls, navgrid.trans_probs)  
        self.anti_sym = navgrid.anti_sym
        self.values = values
        self.newRewards = {}

        # Modify the rewards and transition probabilities to ensure the q-values remain reflected across
        # the diagonal.  We set the rewards to be r_old + gamma[Value_i - Value_j], where i and j are either
        # left and right, or up and down.
        # Then we swap the transition probabilites for up and down, and left and right.
        
        # Calculate the value differences for the relevant states, 
        # and store them to calculate rewards later on
        states = list(itertools.product(range(self.width), range(self.width)))
        for state in states:
            # Ignore walls or states above the diagonal
            if state[0] < state[1] and state not in self.walls:
                s_up, r_up = navgrid.generateReward(state, 'up')
                value_up = values[s_up]
                s_down, r_down = navgrid.generateReward(state, 'up')
                value_down = values[s_down]
                
                self.newRewards[state, 'up'] = value_up-value_down
                self.newRewards[state, 'down'] = value_down-value_up
                
                s_l, r_l = navgrid.generateReward(state, 'left')
                value_l = values[s_l]
                s_r, r_r = navgrid.generateReward(state, 'right')
                value_r = values[s_r]
                
                self.newRewards[state, 'left'] = value_l-value_r
                self.newRewards[state, 'right'] = value_r-value_l
            
        # For states in the lower diagonal, swap trans_probs
        # up <--> down and left left <--> right        
        
        for state in states:
            if state[0] < state[1] and state not in self.walls:
                self.trans_probs[state, 'up']      = [0, 1, 0, 0]
                self.trans_probs[state, 'down']    = [1, 0, 0, 0]
                self.trans_probs[state, 'left']    = [0, 0, 0, 1]
                self.trans_probs[state, 'right']   = [0, 0, 1, 0]

    def generate(self, action, gamma):
        """ Apply the given action to the current state and return
            an (observation, reward) pair.
            (NavGrid, str, int) -> (int, float)
        """
        # Get the direction we go given our action
        a_action = np.random.choice(self.actions, 1, p=self.trans_probs[self.current_pos, action])[0]
        a_dir = self.action_dirs[a_action]
        
        #Clever min-max bounds checking
        pos = self.current_pos
        pos = (min(max(pos[0] + a_dir[0], 0), self.width-1),
               min(max(pos[1] + a_dir[1], 0), self.height-1))
        
        # If we haven't moved into a wall, update the position
        if pos not in self.walls:
            self.current_pos = pos

        # Calculate the new reward using gamma, if we are in the lower diagonal of the grid
        x = 0
        if (self.current_pos, action) in self.newRewards:
            x = self.newRewards[pos,action]
        
        if self.current_pos == self.goal_pos:
            r = 10.0 + gamma * x
        else:
            r = -1.0 + gamma * x
        return (self.pos_to_state(self.current_pos), r)
    
    
    def print_map(self):
        """ Print an ASCII map of the simulation.
            (NavGrid) -> None
        """
        for y in range(self.height):
            for x in range(self.width):
                pos = (x, y)
                if pos == self.current_pos:
                    print("A ", end=''),
                elif pos == self.init_pos:
                    print("S ", end=''),
                elif pos == self.goal_pos:
                    print("G ", end=''),
                elif pos in self.walls:
                    print("X ", end=''),
                else:
                    print("* ", end=''),
            print("") 


class QNavGridReduced(NavGrid):
    """ Takes a NavGrid class and reduces it under a QDP homomorphism.  
        This reduces the grid to the upper diagonal.
        #TODO test using the lower diagonals instead
    """

    def __init__(self, navgrid, newRewards=None):
        """ Make a new NavGridReduced environment.
            (QNavGridReduced, navgrid, dict) -> int
        """
        self.upper_indices = [(i,j) for i in range(navgrid.width) for j in range(navgrid.width) if i >= j]
        if newRewards:
            self.newRewards = newRewards
        else:
            self.newRewards = {}
        super(QNavGridReduced, self).__init__(navgrid.width, len(navgrid.walls), len(navgrid.anti_sym), navgrid.walls, navgrid.trans_probs)
        self.num_states = int(self.height*(self.height+1)/2)
    
    # Calculate the new reward using gamma, if we are in the lower diagonal of the grid
    def generate(self, action, gamma):
        """ Apply the given action to the current state and return
            an (observation, reward) pair.
            (NavGridReduced, str, gamma) -> (int, float)
        """
        # Get the direction we go given our action
        a_action = np.random.choice(self.actions, 1, p=self.trans_probs[self.current_pos, action])[0]
        a_dir = self.action_dirs[a_action]
        
        #Clever min-max bounds checking
        pos = self.current_pos
        pos = (min(max(pos[0] + a_dir[0], 0), self.width-1),
            min(max(pos[1] + a_dir[1], 0), self.height-1))
        
        # If we haven't moved into a wall, update the position
        # (needs to be in the upper_indices as well)
        if pos not in self.walls and pos in self.upper_indices:
            self.current_pos = pos
        
        x = 0
        if (self.current_pos, action) in self.newRewards:
            x = self.newRewards[self.current_pos,action]
        
        if self.current_pos == self.goal_pos:
            r = 10.0 + gamma * x
        else:
            r = -1.0 + gamma * x
        return (self.pos_to_state(self.current_pos), r)

    # Calculate the new reward using gamma, if we are in the lower diagonal of the grid
    def generateReward(self, state, action, gamma):
        """ Apply the given action to the current state and return
            an (observation, reward) pair.
            (NavGridReduced, str, gamma) -> (int, float)
        """
        # Get the direction we go given our action
        a_action = np.random.choice(self.actions, 1, p=self.trans_probs[state, action])[0]
        a_dir = self.action_dirs[a_action]
        
        #Clever min-max bounds checking
        temp_pos = state
        pos = state
        pos = (min(max(pos[0] + a_dir[0], 0), self.width-1),
            min(max(pos[1] + a_dir[1], 0), self.height-1))
        
        # If we haven't moved into a wall, update the position
        # (needs to be in the upper_indices as well)
        if pos not in self.walls and pos in self.upper_indices:
            temp_pos = pos
        
        x = 0
        if (temp_pos, action) in self.newRewards:
            x = self.newRewards[temp_pos,action]
        
        if temp_pos == self.goal_pos:
            r = 10.0 + gamma * x
        else:
            r = -1.0 + gamma * x
        return (temp_pos, r)

    def pos_to_state(self, pos):
        """ Return the index (representing the state) of the current position.
            #TODO: Check if this is fine
            (Environment, (int, int)) -> int
        """
        return self.upper_indices.index(pos)
    
    
    def state_to_pos(self, state):
        """ Return the index (representing the state) of the current position.
            (Environment, (int, int)) -> int
        """
        return (self.upper_indices[state])
    
    
    def print_map(self):
        """ Print an ASCII map of the simulation.
            (NavGrid) -> None
        """
        for y in range(self.height):
            for x in range(self.width):
                pos = (x, y)
                if pos in self.upper_indices:
                    if pos == self.current_pos:
                        print("A ", end=''),
                    elif pos == self.init_pos:
                        print("S ", end=''),
                    elif pos == self.goal_pos:
                        print("G ", end=''),
                    elif pos in self.walls:
                        print("X ", end=''),
                    else:
                        print("* ", end=''),
                else:
                    print("  ", end=''),
            print("") 


class QNavGridReducedLower(NavGrid):
    """ Takes a NavGrid class and reduces it under a QDP homomorphism.  
        This reduces the grid to the lower diagonals.
    """

    def __init__(self, navgrid, newRewards=None):
        """ Make a new NavGridReduced environment.
            (QNavGridReduced, navgrid, dict) -> int
        """
        self.upper_indices = [(i,j) for i in range(navgrid.width) for j in range(navgrid.width) if i <= j]
        if newRewards:
            self.newRewards = newRewards
        else:
            self.newRewards = {}
        super(QNavGridReducedLower, self).__init__(navgrid.width, len(navgrid.walls), len(navgrid.anti_sym), navgrid.walls, navgrid.trans_probs)
        self.num_states = int(self.height*(self.height+1)/2)
    
    # Calculate the new reward using gamma, if we are in the lower diagonal of the grid
    def generate(self, action, gamma):
        """ Apply the given action to the current state and return
            an (observation, reward) pair.
            (NavGridReduced, str, gamma) -> (int, float)
        """
        # Get the direction we go given our action
        a_action = np.random.choice(self.actions, 1, p=self.trans_probs[self.current_pos, action])[0]
        a_dir = self.action_dirs[a_action]
        
        #Clever min-max bounds checking
        pos = self.current_pos
        pos = (min(max(pos[0] + a_dir[0], 0), self.width-1),
            min(max(pos[1] + a_dir[1], 0), self.height-1))
        
        # If we haven't moved into a wall, update the position
        # (needs to be in the upper_indices as well)
        if pos not in self.walls and pos in self.upper_indices:
            self.current_pos = pos
        
        x = 0
        if (self.current_pos, action) in self.newRewards:
            x = self.newRewards[self.current_pos,action]
        
        if self.current_pos == self.goal_pos:
            r = 10.0 + gamma * x
        else:
            r = -1.0 + gamma * x
        return (self.pos_to_state(self.current_pos), r)

    # Calculate the new reward using gamma, if we are in the lower diagonal of the grid
    def generateReward(self, state, action, gamma):
        """ Apply the given action to the current state and return
            an (observation, reward) pair.
            (NavGridReduced, str, gamma) -> (int, float)
        """
        # Get the direction we go given our action
        a_action = np.random.choice(self.actions, 1, p=self.trans_probs[state, action])[0]
        a_dir = self.action_dirs[a_action]
        
        #Clever min-max bounds checking
        temp_pos = state
        pos = state
        pos = (min(max(pos[0] + a_dir[0], 0), self.width-1),
            min(max(pos[1] + a_dir[1], 0), self.height-1))
        
        # If we haven't moved into a wall, update the position
        # (needs to be in the upper_indices as well)
        if pos not in self.walls and pos in self.upper_indices:
            temp_pos = pos
        
        x = 0
        if (temp_pos, action) in self.newRewards:
            x = self.newRewards[temp_pos,action]
        
        if temp_pos == self.goal_pos:
            r = 10.0 + gamma * x
        else:
            r = -1.0 + gamma * x
        return (temp_pos, r)

    def pos_to_state(self, pos):
        """ Return the index (representing the state) of the current position.
            #TODO: Check if this is fine
            (Environment, (int, int)) -> int
        """
        return self.upper_indices.index(pos)
    
    
    def state_to_pos(self, state):
        """ Return the index (representing the state) of the current position.
            (Environment, (int, int)) -> int
        """
        return (self.upper_indices[state])
    
    
    def print_map(self):
        """ Print an ASCII map of the simulation.
            (NavGrid) -> None
        """
        for y in range(self.height):
            for x in range(self.width):
                pos = (x, y)
                if pos in self.upper_indices:
                    if pos == self.current_pos:
                        print("A ", end=''),
                    elif pos == self.init_pos:
                        print("S ", end=''),
                    elif pos == self.goal_pos:
                        print("G ", end=''),
                    elif pos in self.walls:
                        print("X ", end=''),
                    else:
                        print("* ", end=''),
                else:
                    print("  ", end=''),
            print("") 



