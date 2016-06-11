__author__ = 'SohaibY'

# This file contains the code for the inner workings of the multilayer perceptron.

import numpy as np
import scipy.misc
import KTimage as KT


class NeuralNetwork:
    def __init__(self, layerSize):
        # Initializing network using 'layerSize',
        # 'layerSize' is a list containing the number of neurons in input, hidden and output layer

        # Initializing the values of weights randomly
        self.weights_hidden = np.random.normal(loc=0.0, scale=0.1, size=(layerSize[1], layerSize[0]))
        self.weights_output = np.random.normal(loc=0.0, scale=0.1, size=(layerSize[2], layerSize[1]))

        # Initializing the values of biases randomly
        self.bias_hidden = np.random.normal(loc=0.0, scale=0.01, size=(layerSize[1]))
        self.bias_output = np.random.normal(loc=0.0, scale=0.01, size=(layerSize[2]))

        self.layersize = layerSize

    def predict(self, i):
        # Feedforward step using 'i' as values of input layer

        # Calculating values for hidden layer
        self.in_hidden_layer = np.dot(self.weights_hidden, i) + self.bias_hidden
        # Passing those values through activation function
        self.out_hidden_layer = self.tanh(self.in_hidden_layer)

        # Calculating values for output layer
        self.output_layer = np.dot(self.weights_output, self.out_hidden_layer) + self.bias_output

        return self.output_layer

    def train(self, i, e, alpha=0.1):
        # Backpropagation step using 'i' as input of the network, 'e' as backpropagation error
        # and 'alpha' learning rate for backpropagation

        # Feedforward step using input 'i' for calculating the output of the network
        self.predict(i)

        # Setting 'e' as error for output layer
        self.error = e

        # Calculating error for hidden layer
        delta = self.tanh_prime(self.out_hidden_layer) * np.dot(self.weights_output.T, self.error)

        # Updating weights and biases from error
        self.weights_output += alpha * (np.outer(self.error, self.out_hidden_layer))
        self.weights_hidden += alpha * (np.outer(delta, i))
        self.bias_output += alpha * self.error
        self.bias_hidden += alpha * delta

    # Calling external program for visualizing the weights and activations of the network
    def vis(self):
        KT.exporttiles(self.weights_hidden, self.layersize[0], 1, "result/obs_W_1_0.pgm", 1, self.layersize[1])
        KT.exporttiles(self.weights_output, self.layersize[1], 1, "result/obs_W_2_1.pgm", 1, self.layersize[2])


    # Activation functions:
    # Sigmoid function
    def sgm(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    # Inverse sigmoid function
    def sgm_prime(self, x):
        return x * (1 - x)

    # Hyperbolic tangent function
    def tanh(self, x):
        return np.tanh(x)

    # Inverse hyperbolic tangent function
    def tanh_prime(self, x):
        return 1 - (x * x)

    # Sotmax function
    def softmax(self, x):
        return np.exp(x - scipy.misc.logsumexp(x))

    # Inverse softmax function
    def softmax_prime(self, x):
        return (1 - self.softmax(x)) * self.softmax(x)