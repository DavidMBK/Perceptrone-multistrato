import numpy as np
from Utils.MLPNetwork import *
import json
import matplotlib.pyplot as plt
import mplcyberpunk

# Lettura/Utilizzo del Dataset
def Load_Dataset(file_path):
        training_dataset = open(file_path, "r")
        X = [] # Carattersitiche (Misure del fiore)
        Y = [] # Target (Tipologia del fiore)

        for line in training_dataset:
                splitted_line = line.strip().split(",")
                features = list(map(float, splitted_line[:-1]))  
                target = splitted_line[-1]
        
                X.append(features)  
                Y.append(target)

        # Conversione da Stringa ad Intero
        for classes in Y: 
                if classes == "Iris-setosa":
                        Y[Y.index(classes)] = 0
                if classes == "Iris-versicolor":
                        Y[Y.index(classes)] = 1
                if classes == "Iris-virginica":
                        Y[Y.index(classes)] = 2
       
        #print(f"{len(np.unique(Y))} Classi di fiori presenti:", np.unique(Y))
        #print("Dataset utilizzato:", [Y.count(0), Y.count(1), Y.count(2)])
        #print(Y)

        # One hot encoding usato per rappresentare meglio i dati
        # Sennò softmax non saprebbe come lavorare con i dati
        Y = np.eye(len(np.unique(Y)))[Y]  # {1, 0, 0} oppure {0, 1, 0}, etc...     
        # print(Y)

        return X, Y  

# Generalizzazzione del Modello
def Model(X, Y):
        input_size = len(X[0]) # Nel nostro caso sono 4, poiché 1 Neurone di Input = 1 feature
        hidden_size = 5 
        output_size = len(Y[0]) # Qui invece 3, poiché 1 Neurone di Output = 1 classe
        learning_rate = 0.002
        momentum = 0.90
        mlp = MLP(input_size, hidden_size, output_size, learning_rate, momentum)
        return mlp

def get_iterazioni():
        return 100

def export_model(mlp, file_path):
    IrisDataModel = {
        "input_size": mlp.input_size,
        "hidden_size": mlp.hidden_size,
        "output_size": mlp.output_size,
        "learning_rate": mlp.learning_rate,
        "momentum": mlp.momentum,
        
        "Weights": {
            "Input_Hidden_weights": mlp.Input_Hidden_weights.tolist(),
            "Hidden_Output_weights": mlp.Hidden_Output_weights.tolist(),
        },
        "Bias": {
            "Hidden_Layer_bias": mlp.Hidden_Layer_bias.tolist(),
            "Output_Layer_bias": mlp.Output_Layer_bias.tolist(),
        },
        "Momentum": {
            "Input_Hidden_velocity": mlp.Input_Hidden_velocity.tolist(),
            "Hidden_Output_velocity": mlp.Hidden_Output_velocity.tolist(),
            "Hidden_Layer_velocity": mlp.Hidden_Layer_velocity.tolist(),
            "Output_Layer_velocity": mlp.Output_Layer_velocity.tolist(),
        }
    }
    
    with open(file_path, 'w') as file:
        json.dump(IrisDataModel, file, indent=2)
    print(f"Modello Creato in {file_path}")



def import_model(mlp, file_path):
    with open(file_path, 'r') as file:
        IrisDataModel = json.load(file)
    
    mlp.input_size = IrisDataModel["input_size"]
    mlp.hidden_size = IrisDataModel["hidden_size"]
    mlp.output_size = IrisDataModel["output_size"]
    mlp.learning_rate = IrisDataModel["learning_rate"]
    mlp.momentum = IrisDataModel["momentum"]

    mlp.Input_Hidden_weights = np.array(IrisDataModel["Weights"]["Input_Hidden_weights"])
    mlp.Hidden_Output_weights = np.array(IrisDataModel["Weights"]["Hidden_Output_weights"])

    mlp.Hidden_Layer_bias = np.array(IrisDataModel["Bias"]["Hidden_Layer_bias"])
    mlp.Output_Layer_bias = np.array(IrisDataModel["Bias"]["Output_Layer_bias"])
    
    mlp.Input_Hidden_velocity = np.array(IrisDataModel["Momentum"]["Input_Hidden_velocity"])
    mlp.Hidden_Output_velocity = np.array(IrisDataModel["Momentum"]["Hidden_Output_velocity"])
    mlp.Hidden_Layer_velocity = np.array(IrisDataModel["Momentum"]["Hidden_Layer_velocity"])
    mlp.Output_Layer_velocity = np.array(IrisDataModel["Momentum"]["Output_Layer_velocity"])

    print(f"Model imported from {file_path}")