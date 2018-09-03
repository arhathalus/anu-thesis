# Skeleton for this code comes from COMP6320 Artificial Intelligence
# The Australian National University - 2015

"""
Name: Timothy McMahon
Student number: u5530441
"""

"""     
    We suggest using Matplotlib to make plots. Example code to make a
    plot is included below. Matplotlib is installed on the lab computers.
    On Ubuntu and other Debian based distributions of linux, you can install it
    with "sudo apt-get install python-matplotlib".
    
    Please put the code to generate different plotsinto different functions and
    at the bottom of this file put clearly labelled calls to these
    functions, to make it easier to test your generation code.
"""

#TODO Delete unnecesary code
# move to better folder
# make environment and agent
# learn the q-values using q-learning or VI or both and compare
# reduce under homomorphism, learn q values
# find deterministic policy, uplift to bigger domain, check performance

# # learn the q-values using q-learning or VI or both and compare
# construct q-value constant reflection, but break MDPness of homomorphism
# reduce,
# learn,
# test performance

from environment import NavGrid
#from agents import Sarsa, QLearn
import matplotlib.pyplot as plt

def value_iteration(env, gamma, epsilon):
    """ Performs value iteration on an environment, given the parameters gamma and epsilon
        Returns a deterministic policy, and all the values for the states
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




#Here is how to make the environment and agent for Q2 and Q4
#env = get_windy_grid()
#agent = Sarsa(alpha, gamma, epsilon, env.num_states, env.actions)


#Here is how to make the environment and agent for Q3, Q4, Q5
#env = get_windy_grid_water(delta, water_reward)
#agent = qlearn(alpha, gamma, epsilon, env.num_states, env.actions)  


#Here is how to plot with matplotlib
#import matplotlib.pyplot as plt
#plt.title("A test plot")
#plt.xlabel("x label")
#plt.ylabel("y label")
#plt.plot([1,2, 3, 4, 5], [1, 2, 3, 4, 5], label="line1")
#plt.plot([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], label="line2")
#plt.yscale('linear')
#plt.legend(loc='upper left')
#plt.show()

#To use the graphical visualisation system, read the comments in visualisation.py
#This is totally optional and only exists to help you understand what your
#algorithms are doing

#Above your main loop
#import visualisation, time
#vis = visualisation.Visualisation(env, 800, 600, min_reward, max_reward)

#During an episode build up trace [(state, action)]
#At the ends of an episode do
#vis.show_Q(agent, show_greedy_policy, trace)
#time.sleep(0.1)

#At the end block until the visualisation is closed.
#vis.pause()


def question2_learning_rates():

    gamma = 0.99
    epsilon = 0.01
    alphas = [0.2, 0.4, 0.6, 0.8, 1]

    for alpha in alphas:

        env = get_windy_grid()
        agent = Sarsa(alpha, gamma, epsilon, env.num_states, env.actions)
        time = 0
        ep_count = 0
        time_steps = [0]
        episode_end = [0]

        while time < 8000:

            action = agent.select_action(env.init_state)
            state = env.init_state

            while not env.end_of_episode():

                new_state, reward = env.generate(action)
                time += 1
                new_action = agent.select_action(new_state)

                agent.update_Q(state, action, reward, new_state, new_action)

                state = new_state
                action = new_action

            ep_count += 1
            time_steps.append(time)
            episode_end.append(ep_count)

        plt.title("Sarsa Learning Rates")
        plt.xlabel("Time Steps")
        plt.ylabel("Episodes")
        plt.plot(time_steps, episode_end, label="alpha = " + str(alpha))

    plt.yscale('linear')
    plt.legend(loc='upper left')
    plt.show()


def question3_stochastic_winds():

    water_reward = -10
    alpha = 0.1
    epsilon = 0.005
    gamma = 0.99

    delta_array = []
    episode_length = []
    deltas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for delta in deltas:

        env = get_windy_grid_water(delta, water_reward)
        agent = Sarsa(alpha, gamma, epsilon, env.num_states, env.actions)
        time = 0

        for episode in range(100000):

            action = agent.select_action(env.init_state)
            state = env.init_state

            while not env.end_of_episode():

                new_state, reward = env.generate(action)
                time += 1
                new_action = agent.select_action(new_state)

                agent.update_Q(state, action, reward, new_state, new_action)

                state = new_state
                action = new_action

        ep_length = 0
        agent.epsilon = 0

        for episode in range(500):

            action = agent.select_action(env.init_state)
            state = env.init_state

            while not env.end_of_episode():

                new_state, reward = env.generate(action)
                ep_length += 1
                new_action = agent.select_action(new_state)

                agent.update_Q(state, action, reward, new_state, new_action)

                state = new_state
                action = new_action

        delta_array.append(delta)
        episode_length.append(ep_length/500.0)

        print "delta" + str(delta)
        print ep_length / 500.0

        plt.title("Stochastic Winds")
        plt.xlabel("Probability of wind")
        plt.ylabel("Episode Length")
        plt.plot(delta_array, episode_length, label="delta = " + str(alpha))

    plt.yscale('linear')
    plt.show()


def question4_q_learn_sarsa():

    water_reward = -100
    alpha = 0.1
    gamma = 0.99
    delta = 0

    epsilon_array = []

    queue_episode_length = []
    sarsa_episode_length = []

    queue_average_score = []
    sarsa_average_score = []


    # Flag to set epsilon to zero after convergence
    #epsilon_to_null = False
    epsilon_to_null = True

    epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for epsilon in epsilons:

        # QUEUE LEARNING
        env = get_windy_grid_water(delta, water_reward)
        agent = QLearn(alpha, gamma, epsilon, env.num_states, env.actions)
        ep_count = 0
        for episode in range(100000):

            state = env.init_state

            while not env.end_of_episode():

                action = agent.select_action(state)
                new_state, reward = env.generate(action)

                agent.update_Q(state, action, reward, new_state)

                state = new_state

        totals_reward = 0
        ep_length = 0

        # Turn Epsilon off
        if epsilon_to_null:
            agent.epsilon = 0

        for episode in range(500):

            state = env.init_state

            while not env.end_of_episode():

                action = agent.select_action(state)
                new_state, reward = env.generate(action)
                print new_state
                ep_length += 1
                totals_reward += reward

                agent.update_Q(state, action, reward, new_state)
                state = new_state

        epsilon_array.append(epsilon)
        queue_average_score.append(totals_reward/500.0)
        queue_episode_length.append(ep_length/500.0)

        print " Queue epsilon " + str(epsilon) + " length: " + str(ep_length / 500.0) + " reward " + str(totals_reward / 500.0)

        # SARSA
        agent = Sarsa(alpha, gamma, epsilon, env.num_states, env.actions)

        for episode in range(100000):

            action = agent.select_action(env.init_state)
            state = env.init_state

            while not env.end_of_episode():

                new_state, reward = env.generate(action)
                new_action = agent.select_action(new_state)

                agent.update_Q(state, action, reward, new_state, new_action)

                state = new_state
                action = new_action

        totals_reward = 0
        ep_length = 0

        if epsilon_to_null:
            agent.epsilon = 0

        for episode in range(500):

            action = agent.select_action(env.init_state)
            state = env.init_state

            while not env.end_of_episode():

                new_state, reward = env.generate(action)
                print new_state
                ep_length += 1
                totals_reward += reward
                new_action = agent.select_action(new_state)

                agent.update_Q(state, action, reward, new_state, new_action)

                state = new_state
                action = new_action

        sarsa_episode_length.append(ep_length/500.0)
        sarsa_average_score.append(totals_reward/500.0)

        print " SARSA epsilon " + str(epsilon) + " length: " + str(ep_length / 500.0) + " reward " + str(totals_reward / 500.0)
        # print "epsilon" + str(epsilon)
        # print ep_length / 500.0
        # print totals_reward / 500.0

    if epsilon_to_null:
        epsilon_string = "off"
    else:
        epsilon_string = "on"
    plt.title("Q-learning vs Sarsa with epsilon " + epsilon_string)
    plt.xlabel("Epsilon")
    plt.ylabel("Average Episode Length")
    plt.plot(epsilon_array, queue_episode_length, label="QLearning")
    plt.plot(epsilon_array, sarsa_episode_length, label="Sarsa")
    #plt.plot(epsilon_array, average_score, label="epsilon = " + str(alpha))
    plt.legend(loc='upper left')
    plt.yscale('linear')
    plt.show()

    plt.title("Q-learning vs Sarsa with epsilon " + epsilon_string)
    plt.xlabel("Epsilon")
    plt.ylabel("Average Result")
    plt.plot(epsilon_array, queue_average_score, label="QLearning")
    plt.plot(epsilon_array, sarsa_average_score, label="Sarsa")
    plt.legend(loc='upper left')
    plt.yscale('linear')
    plt.show()


def question_5():

    water_reward = -10
    alpha = 0.1
    epsilon = 0.0
    gamma = 0.99

    delta = 0

    env = get_windy_grid_water(delta, water_reward)
    agent = Sarsa(alpha, gamma, epsilon, env.num_states, env.actions)
    time = 0

    agent.Q = dict([((s, a), -100.0) for s in xrange(agent.num_states) for a in agent.actions])

    action = agent.select_action(env.init_state)
    state = env.init_state

    while time < 500000:

        new_state, reward = env.generate(action)
        time += 1
        print new_state, reward
        new_action = agent.select_action(new_state)

        agent.update_Q(state, action, reward, new_state, new_action)

        state = new_state
        action = new_action

        if env.end_of_episode():
            print "Success"
            break

if __name__ == '__main__':
    #question4_q_learn_sarsa()
    #question_5()
    question3_stochastic_winds()