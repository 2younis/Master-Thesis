__author__ = 'SohaibY'

# This file contains the code for reinforcement learning using recurrent neural network

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

# Importing the code for taxi environment and multilayer perceptron
import taxi_environment as taxi_environment
import mlp as mlp


class learning():
    def __init__(self, alpha=0.1, gamma=0.9):
        # Initializing the Q-learning agent by using learning rate 'alpha' and discount factor 'gamma'

        # Creating taxi environment
        self.environment = taxi_environment.environment()

        # Creating a dictionary for possible actions
        self.options = {0: 'n', 1: 's', 2: 'e', 3: 'w', 4: 'p', 5: 'd'}

        # Setting the values for alpha and gamma
        self.alpha = alpha
        self.gamma = gamma

        # Initializing the value for tau
        self.tau = 0.0

        # Setting the number of neurons in high-level part of output layer
        self.sub_goal_no = 4
        # Initializing the values of these neurons
        self.sub_goal = np.zeros(self.sub_goal_no)

        # Initializing the histogram of activations of high level neurons of output layer for each low level neuron
        self.freq_action = np.zeros((6, self.sub_goal_no))

        # initializing the value of previous action
        self.prv_action = np.zeros(6)

        # Retrieving the latest states of the environment
        self.update_states()

        # Creating multilayer perceptron by defining the number of neurons in input, hidden and output layers
        # The output layer consists of standard actions and high level actions
        # The input layer consists of environment states and feedback of output layer
        self.network = mlp.NeuralNetwork([len(self.states), 60, 6 + self.sub_goal_no])

    def select_action(self, q):
        # Selecting an action with Boltzmann action selection using 'q' as Q-values

        # Calculating temperature for Boltzmann action selection
        temp = 1 / float(self.environment.epoch + 1)

        # Setting lower limit for temperature
        if temp < 0.05:
            temp = 0.05

        # Calculating the probabilities of the actions using Q-values
        div_prob = q / temp
        prob = np.exp(div_prob - scipy.misc.logsumexp(div_prob))

        # Selecting an action based on its probability
        cs = np.cumsum(prob)
        idx = np.sum(cs < np.random.rand())
        return idx

    # @property
    def update_states(self):

        # Reset each neuron representing pick up and drop off location
        # The total number of neurons depend of the number of total available pick up and drop off locations
        pick_up = np.zeros(self.environment.goals)
        drop_off = np.zeros(self.environment.goals)

        # Set the value of the neuron representing the current pick up and drop off location, respectively
        pick_up[self.environment.pickloc_no] = 1.0
        drop_off[self.environment.droploc_no] = 1.0

        # Retrieve the latest position of the taxi from the environment
        self.states = np.array([self.environment.taxi_pos[0], self.environment.taxi_pos[1]])
        # Retrieve the current pick up and drop off locations
        self.states = np.concatenate((self.states, pick_up, drop_off), axis=0)
        # Retrieve the current value of the passenger
        self.states = np.append(self.states, self.environment.passenger)

        # Retrieve the previous (feedback) values of high level part of output layer
        self.states = np.concatenate((self.states, self.sub_goal), axis=0)
        # Retrieve the previous (feedback) value of low level part of output layer
        self.states = np.concatenate((self.states, self.prv_action), axis=0)

        return self.states

    def update_q(self):
        # Updating the Q-values with Q-learning

        # Retrieve the current state of the environment
        current_state = self.update_states()
        # Give this current state to the network as input to approximate the Q-values of the actions
        Q = self.network.predict(current_state)

        # Separate the Q-values of high level and low level part of output
        l_Q = Q[:6]
        h_Q = Q[6:]

        # Select an action using Q-values approximated by the network for each part of output
        l_action = self.select_action(l_Q)
        h_action = self.select_action(h_Q)

        # Reset the feedback values from previous output
        self.sub_goal = np.zeros(self.sub_goal_no)
        self.prv_action = np.zeros(6)

        # Set the feedback value for high level part of output layer
        self.sub_goal[h_action] = 1.0

        # Increment the value of tau
        self.tau += 1.0

        # Increment the frequency for histogram after 30000 epochs
        if self.environment.epoch > 30000:
            self.freq_action[l_action][h_action] += 1.0

        # Set the feedback value for low level part of output layer
        self.prv_action[l_action] = 1.0

        # Implement the selected low level action in the environment
        self.environment.navigate(self.options[l_action])

        # Retrieve the new state of the environment
        new_state = self.update_states()
        # Find the Q-values for the new state
        new_Q = self.network.predict(new_state)

        # Select the maximum Q-values of new state for each part of output layer
        l_maxQ = max(new_Q[:6])
        h_maxQ = max(new_Q[6:])

        # Initialize the backpropagation error for the network
        deltaQ = np.zeros(len(Q))

        # Update the Q-value of the low level selected action using temporal difference
        deltaQ[l_action] = self.alpha * (self.environment.reward + (self.gamma * l_maxQ) - Q[l_action])

        # Update the Q-value of the high level selected action using temporal difference
        # only when the agent receives any reward
        if self.environment.reward > 0.0 and self.environment.epoch > 0:
            deltaQ[h_action + 6] = (self.alpha * 1e-5) * ((self.environment.reward ** self.tau) +
                                                          ((self.gamma ** self.tau) * h_maxQ) - Q[h_action])
            # Reset the value of tau
            self.tau = 0.0

        # Provide the network with current state and error for backpropagation in order to update its weights
        self.network.train(current_state, deltaQ)

        # Update the position of the taxi in the environment
        self.environment.update_pos()


if __name__ == "__main__":

    learn = learning()

    it = 0  # Counter for steps or actions taken by the agent so far in this epoch
    avgint = 0  # Counter for total number of steps taken by the agent within the current 50 epoch span
    avg = 0  # Average number of steps taken by the agent in the last 50 epoch span
    c_epoch = 1  # Current epoch counter
    total_epochs = 50000  # Total number of epochs allowed for the agent

    # Initialize empty lists for plotting the average number of steps taken by the agent at each (50) epoch
    epochs = []
    average = []

    while learn.environment.epoch < total_epochs:
        # Perform Q-learning update step
        learn.update_q()

        it += 1
        avgint += 1

        # Print the current epoch number and the number of steps taken by the agent in this epoch so far
        print learn.environment.epoch, it

        if (learn.environment.epoch % 50 == 0) and learn.environment.epoch > c_epoch:
            # Calculate average number of steps taken at the end of each 50 epochs
            avg = avgint / 50
            avgint = 0

        if c_epoch < learn.environment.epoch:
            # Append new values of average for plotting
            epochs.append(learn.environment.epoch - 2)
            average.append(avg)
            # Increment current epoch counter
            c_epoch += 1
            # Update counter for steps
            it = 0

    # Visualize the current weights and activations of the network
    learn.network.vis()

    # Plot the average number of steps taken by the agent at each (50) epoch
    plt.figure(1)
    plt.axis([0, total_epochs, 0, 200])
    plt.ion()
    plt.show()
    plt.grid()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    plt.xlabel(r"episodes", fontsize=20)
    plt.ylabel(r"average number of steps", fontsize=20)
    plt.plot(epochs, average, 'r.')
    plt.draw()
    plt.savefig('average_recurrent_ql.pdf', format='pdf', dpi=400)

    # Plot the histogram of activations of high level neurons of output layer for each low level neuron
    fig2 = plt.figure(2)
    ax = fig2.add_subplot(111, projection='3d')
    for c, z in zip(['r', 'g', 'b', 'y'], [0, 1, 2, 3]):
        cs = [c] * 6
        ax.bar(np.arange(6), learn.freq_action.T[:][z], z, zdir='y', color=cs, alpha=0.8)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tick_params(axis='both', which='minor', labelsize=8)
    ax.set_xlabel(r"low-level action", fontsize=14)
    ax.set_ylabel(r"high-level action", fontsize=14)
    ax.set_zlabel(r"frequency", fontsize=14)
    fig2.savefig('freq_ac.pdf', format='pdf')