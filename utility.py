import numpy as np
from MLPNetwork import *

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
       
        print("Test dei valori:", np.unique(Y))
        print("Come sono distribuiti:", [Y.count(0), Y.count(1), Y.count(2)])
        
        Y = np.eye(len(np.unique(Y)))[Y] # One hot encoding

        return X, Y  

# Generalizzazzione del Modello
def Model(X, Y):
        input_size = len(X[0]) # Nel nostro caso sono 4, poiché 1 Neurone di Input = 1 feature
        hidden_size = 8 
        output_size = len(Y[0]) # Qui invece 3, poiché 1 Neurone di Output = 1 classe
        learning_rate = 0.01
        momentum = 0.9
        mlp = MLP(input_size, hidden_size, output_size, learning_rate, momentum)
        return mlp