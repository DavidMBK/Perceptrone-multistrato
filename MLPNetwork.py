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
        self.IH_weights = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
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
    
    def sigmoid_derivative(self, x):
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
        output_delta = output_error * self.sigmoid_derivative(self.output_input)

        # get error at hidden layer
        hidden_error = np.dot(output_delta, self.HO_weights.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_input)

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
    
    def export_model(self, file_path):
        model_data = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "IH_weights": self.IH_weights.tolist(),
            "HO_weights": self.HO_weights.tolist(),
            "HL_bias": self.HL_bias.tolist(),
            "OL_bias": self.OL_bias.tolist(),
            "IH_velocity": self.IH_velocity.tolist(),
            "HO_velocity": self.HO_velocity.tolist(),
            "HL_velocity": self.HL_velocity.tolist(),
            "OL_velocity": self.OL_velocity.tolist()
        }
        
        with open(file_path, 'w') as file:
            json.dump(model_data, file, indent=4)
        print(f"Model exported to {file_path}")


    def import_model(self, file_path):
        with open(file_path, 'r') as file:
            model_data = json.load(file)

        self.input_size = model_data["input_size"]
        self.hidden_size = model_data["hidden_size"]
        self.output_size = model_data["output_size"]
        self.learning_rate = model_data["learning_rate"]
        self.momentum = model_data["momentum"]

        self.IH_weights = np.array(model_data["IH_weights"])
        self.HO_weights = np.array(model_data["HO_weights"])
        self.HL_bias = np.array(model_data["HL_bias"])
        self.OL_bias = np.array(model_data["OL_bias"])
        self.IH_velocity = np.array(model_data["IH_velocity"])
        self.HO_velocity = np.array(model_data["HO_velocity"])
        self.HL_velocity = np.array(model_data["HL_velocity"])
        self.OL_velocity = np.array(model_data["OL_velocity"])

        print(f"Model imported from {file_path}")