__author__ = 'SohaibY'

import numpy as np
import decimal as dc
import scipy.misc
import matplotlib.pyplot as plt

import stage_taxi_environment as taxi_environment
import mlp as mlp


class learning():
    def __init__(self, alpha=0.1, gamma=0.9):
        # Initializing the Q-learning agent by using learning rate 'alpha' and discount factor 'gamma'

        # Creating taxi environment
        self.environment = taxi_environment.environment()

        # Creating multilayer perceptrons by defining the number of neurons in input, hidden and output layers
        # Creating low level multilayer perceptron network
        self.network1 = mlp.NeuralNetwork([4, 30, 6])
        # Creating high level multilayer perceptron network
        self.network2 = mlp.NeuralNetwork([3, 20, 4])

        # Creating a dictionary for possible actions
        self.options = {0: 'n', 1: 's', 2: 'e', 3: 'w', 4: 'p', 5: 'd'}

        # Setting the value for the total number of epochs for the first stage
        self.environment.first_stage_epochs = 25000

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

    def update_states(self, index=1):

        # Retrieving the latest states of the environment as input relevant for the low level network
        self.states1 = np.array([self.environment.taxi_pos[0], self.environment.taxi_pos[1], self.environment.goal_no,
                                 self.environment.passenger])
        # Retrieving the latest states of the environment as input relevant for the high level network
        self.states2 = np.array([self.environment.passenger, self.environment.pickloc_no, self.environment.droploc_no])
        if index == 1:
            return self.states1
        elif index == 2:
            return self.states2

    def update_nav(self):
        # Updating the Q-values with Q-learning for the first stage

        # Retrieve the current state of the environment for low level network
        current_state = self.update_states(1)
        # Give this current state to the low level network as input to approximate the Q-values of the actions
        Q1 = self.network1.predict(current_state)

        # Select an action using Q-values approximated by the network
        current_action = self.select_action(Q1)
        # Implement the selected action in the environment
        self.environment.navigate(self.options[current_action])

        # Retrieve the new state of the environment for the low level network
        new_state = self.update_states(1)
        # Select the maximum Q-value from output of the low level network using new state as input
        maxQ1 = max(self.network1.predict(new_state))

        # Initialize the backpropagation error for the low level network
        deltaQ1 = np.zeros(len(Q1))

        # Update the Q-value of the selected action using temporal difference
        deltaQ1[current_action] = self.alpha * (self.environment.reward + (self.gamma * maxQ1) - Q1[current_action])

        # Provide the low level network with current state and error for backpropagation in order to update its weights
        self.network1.train(current_state, deltaQ1)

        # Update the position of the taxi in the environment
        self.environment.update_pos()

    def update_taxi(self):

        # If agent is in the first stage of learning
        if self.environment.epoch < self.environment.first_stage_epochs:
            # Update Q-vales for first stage
            self.update_nav()
        else:
            # Updating the Q-values with Q-learning for the second stage

            # Retrieve the current state of the environment for high level network
            c_state = self.update_states(2)
            # Give this current state to the high level network as input to approximate the Q-values of the actions
            Q2 = self.network2.predict(c_state)
            # Select an action using Q-values approximated by the network
            c_action = self.select_action(Q2)
            # Set the selected action as current goal position
            self.environment.goal_no = c_action

            # Retrieve the current state of the environment for low level network
            current_state = self.update_states(1)
            # Give this current state to the low level network as input to approximate the Q-values of the actions
            Q1 = self.network1.predict(current_state)

            # Select an action using Q-values approximated by the network
            current_action = self.select_action(Q1)
            # Implement the selected action in the environment
            self.environment.navigate(self.options[current_action])

            # Retrieve the new state of the environment for the high level network
            n_state = self.update_states(2)
            # Select the maximum Q-value from output of the high level network using new state as input
            maxQ2 = max(self.network2.predict(n_state))

            # Initialize the backpropagation error for the high level network
            deltaQ2 = np.zeros(len(Q2))

            # Update the Q-value of the selected action using temporal difference
            deltaQ2[c_action] = (self.alpha * 1e-3) * ((self.environment.reward) + (self.gamma * maxQ2) - Q2[c_action])

            # Provide the low level network with current state and error for backpropagation
            # in order to update its weights
            self.network2.train(self.states2, deltaQ2)

            # Update the position of the taxi in the environment
            self.environment.update_pos()


if __name__ == "__main__":

    learn = learning()

    it = 0  # Counter for steps or actions taken by the agent so far in this epoch
    avgint = 0  # Counter for total number of steps taken by the agent within the current 50 epoch span
    avg = 0  # Average number of steps taken by the agent in the last 50 epoch span
    c_epoch = 1  # Current epoch counter
    total_epochs = 50000  # Total number of epochs allowed for the agent
    learn.environment.first_stage_epochs = 25000  # The value for the total number of epochs for the first stage


    # Initialize empty lists for plotting the average number of steps taken by the agent at each (50) epoch
    epochs = []
    average = []

    while learn.environment.epoch < total_epochs:
        # Perform Q-learning update step
        learn.update_taxi()
        it += 1
        avgint += 1

        # Print the current epoch number and the number of steps taken by the agent in this epoch so far
        print learn.environment.epoch, it

        # Reset the environment if the number of steps exceed the upper limit in an epoch
        if it > 500:
            learn.environment.reset()
            it = 0

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

    # Plot the average number of steps taken by the agent at each (50) epoch
    plt.figure(1)
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
    plt.savefig('modular_hql_average.pdf', format='pdf', dpi=400)
