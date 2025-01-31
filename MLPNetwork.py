import numpy as np
import json
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, momentum=0.9):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        # initialize neurons
        self.input_neurons = np.zeros(self.input_size)
        self.hidden_neurons = np.zeros(self.hidden_size)
        self.output_neurons = np.zeros(self.output_size)

        # weights for input -> hidden and for hidden -> output
        self.InputH_weights = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.HO_weights = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        
        # bias for hidden and output layer
        self.HL_bias = np.zeros((1, self.hidden_size))
        self.OL_bias = np.zeros((1, self.output_size))
    
        # velocity terms for momentum
        self.IH_velocity = np.zeros_like(self.IH_weights)
        self.HO_velocity = np.zeros_like(self.HO_weights)
        self.HL_velocity = np.zeros_like(self.HL_bias)
        self.OL_velocity = np.zeros_like(self.OL_bias)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoide_derivate(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def feedforward(self, X):
        # input -> hidden
        self.input_neurons = X
        self.hidden_input = np.dot(self.input_neurons, self.IH_weights) + self.HL_bias
        self.hidden_neurons = self.sigmoid(self.hidden_input)

        # hidden -> output
        self.output_input = np.dot(self.hidden_neurons, self.HO_weights) + self.OL_bias
        self.output_neurons = self.sigmoid(self.output_input)

        return self.output_neurons

    def backpropagation(self, X, y):
        # get error at output layer
        output_error = self.output_neurons - y
        output_delta = output_error * self.sigmoide_derivate(self.output_input)

        # get error at hidden layer
        hidden_error = np.dot(output_delta, self.HO_weights.T)
        hidden_delta = hidden_error * self.sigmoide_derivate(self.hidden_input)

        # update weights and biases with momentum
        self.HO_velocity = self.momentum * self.HO_velocity - self.learning_rate * np.dot(self.hidden_neurons.T, output_delta)
        self.HO_weights += self.HO_velocity

        self.OL_velocity = self.momentum * self.OL_velocity - self.learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.OL_bias += self.OL_velocity

        self.IH_velocity = self.momentum * self.IH_velocity - self.learning_rate * np.dot(X.T, hidden_delta)
        self.IH_weights += self.IH_velocity

        self.HL_velocity = self.momentum * self.HL_velocity - self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)
        self.HL_bias += self.HL_velocity

    def get_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)