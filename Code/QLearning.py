__author__ = 'SohaibY'

# This file contains the code for reinforcement learning using multilayer perceptron

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt


# Importing the code for taxi environment and multilayer perceptron
import taxi_environment
import mlp


class learning():
    def __init__(self, alpha=0.1, gamma=0.9):
        # Initializing the Q-learning agent by using learning rate 'alpha' and discount factor 'gamma'

        # Creating taxi environment
        self.environment = taxi_environment.environment()
        # Creating multilayer perceptron by defining the number of neurons in input, hidden and output layers
        self.network = mlp.NeuralNetwork([5, 30, 6])

        # Creating a dictionary for possible actions
        self.options = {0: 'n', 1: 's', 2: 'e', 3: 'w', 4: 'p', 5: 'd'}

        # Setting the values for alpha and gamma
        self.alpha = alpha
        self.gamma = gamma

        # Retrieving the latest states of the environment
        self.update_states()

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

    def update_states(self):
        # Retrieving the latest states of the environment as input for the network
        self.states = np.array([self.environment.taxi_pos[0], self.environment.taxi_pos[1],
                                self.environment.passenger, self.environment.pickloc_no, self.environment.droploc_no])
        return self.states

    def update_q(self):
        # Updating the Q-values with Q-learning

        # Retrieve the current state of the environment
        current_state = self.update_states()
        # Give this current state to the network as input to approximate the Q-values of the actions
        Q = self.network.predict(current_state)

        # Select an action using Q-values approximated by the network
        current_action = self.select_action(Q)
        # Implement the selected action in the environment
        self.environment.navigate(self.options[current_action])

        # Retrieve the new state of the environment
        new_state = self.update_states()
        # Select the maximum Q-value from output of the network using new state as input
        maxQ = max(self.network.predict(new_state))

        # Initialize the backpropagation error for the network
        deltaQ = np.zeros(len(Q))

        # Update the Q-value of the selected action using temporal difference
        deltaQ[current_action] = self.alpha * (self.environment.reward + (self.gamma * maxQ) - Q[current_action])

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
    plt.savefig('average_ql.pdf', format='pdf', dpi=400)
