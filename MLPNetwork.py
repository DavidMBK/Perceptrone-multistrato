import numpy as np
import json
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, momentum):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        # initialize neurons
        self.input_neurons = np.zeros(self.input_size)
        self.hidden_neurons = np.zeros(self.hidden_size)
        self.output_neurons = np.zeros(self.output_size)

        # [Weights]
        self.Input_Hidden_weights = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
        self.Hidden_Output_weights = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
        
        # [Bias]
        self.Hidden_Layer_bias = np.zeros((1, self.hidden_size))
        self.Output_Layer_bias = np.zeros((1, self.output_size))
    
        # [Momentum]
        self.Input_Hidden_velocity = np.zeros_like(self.Input_Hidden_weights)
        self.Hidden_Output_velocity = np.zeros_like(self.Hidden_Output_weights)
        self.Hidden_Layer_velocity = np.zeros_like(self.Hidden_Layer_bias)
        self.Output_Layer_velocity = np.zeros_like(self.Output_Layer_bias)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoide_derivate(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def feedforward(self, X):
        # input -> hidden
        self.input_neurons = X
        self.hidden_input = np.dot(self.input_neurons, self.Input_Hidden_weights) + self.Hidden_Layer_bias
        self.hidden_neurons = self.sigmoid(self.hidden_input)

        # hidden -> output
        self.output_input = np.dot(self.hidden_neurons, self.Hidden_Output_weights) + self.Output_Layer_bias
        self.output_neurons = self.sigmoid(self.output_input)

        return self.output_neurons

    def backpropagation(self, X, y):
        # get error at output layer
        output_error = self.output_neurons - y
        output_delta = output_error * self.sigmoide_derivate(self.output_input)

        # get error at hidden layer
        hidden_error = np.dot(output_delta, self.Hidden_Output_weights.T)
        hidden_delta = hidden_error * self.sigmoide_derivate(self.hidden_input)

        # update weights and biases with momentum
        self.Hidden_Output_velocity = self.momentum * self.Hidden_Output_velocity - self.learning_rate * np.dot(self.hidden_neurons.T, output_delta)
        self.Hidden_Output_weights += self.Hidden_Output_velocity

        self.Output_Layer_velocity = self.momentum * self.Output_Layer_velocity - self.learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.Output_Layer_bias += self.Output_Layer_velocity

        self.Input_Hidden_velocity = self.momentum * self.Input_Hidden_velocity - self.learning_rate * np.dot(X.T, hidden_delta)
        self.Input_Hidden_weights += self.Input_Hidden_velocity

        self.Hidden_Layer_velocity = self.momentum * self.Hidden_Layer_velocity - self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)
        self.Hidden_Layer_bias += self.Hidden_Layer_velocity

    def get_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)