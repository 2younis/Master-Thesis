__author__ = 'SohaibY'

# This file contains the code for inner working of the taxi environment for two stage learning.

import random as rand


class environment:
    def __init__(self):

        # Initializing the grid size for the taxi environment
        self.grid_size = 5

        # Defining the available pick up and drop off locations
        self.loc = []
        self.loc.append([4, 4])
        self.loc.append([3, 4])
        self.loc.append([2, 2])
        self.loc.append([3, 0])

        # Initializing counter for epochs
        self.epoch = 1  # integer

        # Setting initial values for reward, passenger and success
        self.passenger = 0  # boolean
        self.reward = 0  # float
        self.success = 0  # boolean

        # Randomizing the initial position of the taxi
        self.taxi_pos = [rand.randrange(5), rand.randrange(5)]

        # Setting the value for the total number of epochs for the first stage
        self.first_stage_epochs = 25000

        # Initializing the pick up and drop off locations
        self.reset()

    def reset(self):

        # If agent is in the first stage of learning
        if self.epoch < self.first_stage_epochs:
            # Randomly select the  pick up and drop off location from predefined available locations
            self.pickloc_no = rand.randrange(4)
            self.droploc_no = rand.randrange(4)
            self.pickup = self.loc[self.pickloc_no]
            self.drop_off = self.loc[self.droploc_no]

            # Set the pick up location as the current goal position
            self.goal_no = self.pickloc_no
            self.goal = self.pickup

            # Reset the value for passenger
            self.passenger = 0
        else:
            # Randomly select the  pick up and drop off location from predefined available locations
            self.pickloc_no = rand.randrange(4)
            self.droploc_no = rand.randrange(4)
            self.pickup = self.loc[self.pickloc_no]
            self.drop_off = self.loc[self.droploc_no]

            # Reset the value for passenger
            self.passenger = 0

    def update_pos(self):

        # Resetting the value of reward in case the agent had previously received reward
        if self.reward > 0:
            self.reward = 0

        # If the agent was successful in picking up the passenger and dropping it off correctly
        if self.success == 1:
            # Reset the value for success
            self.success = 0
            # Reset the position of the taxi randomly
            self.taxi_pos = [rand.randrange(5), rand.randrange(5)]

    def navigate(self, action):

        # If the selected action is move 'north' and the taxi is within the grid
        if action == 'n' and self.taxi_pos[1] < 4:
            # Move the taxi north by one unit
            self.taxi_pos[1] += 1

        # If the selected action is move 'south' and the taxi is within the grid
        elif action == 's' and self.taxi_pos[1] > 0:
            # Move the taxi south by one unit
            self.taxi_pos[1] -= 1

        # If the selected action is move 'east' and the taxi is within the grid
        elif action == 'e' and self.taxi_pos[0] < 4:
            # Move the taxi east by one unit
            self.taxi_pos[0] += 1

        # If the selected action is move 'west' and the taxi is within the grid
        elif action == 'w' and self.taxi_pos[0] > 0:
            # Move the taxi west by one unit
            self.taxi_pos[0] -= 1

        # If the selected action is perform 'pick'
        elif action == 'p':
            # Perform pick up action
            self.pick()

        # If the selected action is perform 'drop'
        elif action == 'd':
            # Perform drop off action
            self.drop()

        # If taxi has reached the current goal
        if self.taxi_pos == self.goal:
            # Give reward to the agent
            self.reward += 0.0001

    def pick(self):
        # If the taxi is at the pick up location and the passenger is not on board
        if not self.passenger and self.taxi_pos == self.goal:
            # Pick up the passenger
            self.passenger = 1

            # If agent is in the first stage of learning
            if self.epoch < self.first_stage_epochs:
                # Give reward to the agent
                self.reward += 0.1
                # Set drop off location as current goal location
                self.goal_no = self.droploc_no
            else:
                # Give reward to the agent
                self.reward += 0.01
                # Set drop off location as current goal location
                self.goal_no = self.droploc_no

    def drop(self):
        # If the taxi is at the drop off location and the passenger is on board
        if self.passenger and self.taxi_pos == self.drop_off:
            # Drop off the passenger
            self.passenger = 0
            # Giver reward to the agent
            self.reward += 1
            # Set the value for 'success' as the passenger was successfully dropped of at the correct position
            self.success = 1
            # Increment the counter for epoch
            self.epoch += 1
            # Reset the pick up and drop off locations for next epoch
            self.reset()