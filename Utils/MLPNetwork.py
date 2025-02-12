import numpy as np

def starting_weights (row, col):
    # [Vogliamo che i pesi iniziali non sono uniformi, ma che oscillano tra -1 a 1]
    return np.random.uniform(-1, 1, (row, col))

def starting_bias(matrix):
    # [Bias a 2 dimensioni usato per i calcoli]
    return np.zeros((1, matrix))

def matrix_mul(m1, m2):
    return np.dot(m1, m2)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, momentum):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        # Inizializzazione dei neuroni 
        self.input_neurons = np.zeros(self.input_size)
        self.hidden_neurons = np.zeros(self.hidden_size)
        self.output_neurons = np.zeros(self.output_size)

        # Pesi
        self.Input_Hidden_weights = starting_weights(self.input_size, self.hidden_size)
        self.Hidden_Output_weights = starting_weights(self.hidden_size, self.output_size)
        
        # Bias
        self.Hidden_Layer_bias = starting_bias(self.hidden_size)
        self.Output_Layer_bias = starting_bias(self.output_size)
    
        # Velocità Input - Hidden - Output
        self.Input_Hidden_velocity = np.zeros_like(self.Input_Hidden_weights)
        self.Hidden_Output_velocity = np.zeros_like(self.Hidden_Output_weights)

        # Velocità Layer Hidden - Output 
        self.Hidden_Layer_velocity = np.zeros_like(self.Hidden_Layer_bias)
        self.Output_Layer_velocity = np.zeros_like(self.Output_Layer_bias)

    # Attivazione Sigmoidale per l'hidden
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Attivazione per l'output
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) # Axis serve per indicare le operazioni lungo le righe
        return exp_x / np.sum(exp_x, axis=1, keepdims=True) # Keepdims = True per restituire array in forma (n, 1)

    # Propagazione della rete neurale, viene usato per indicare che 
    # la rete deve non può andare indietro e può andare solo avanti, tramite
    # il formato Input - Hidden - Output dove ogni output è l'input per lo strato successivo.

    def feedforward(self, X):
        self.input_neurons = X # I valori
        self.hidden_input = matrix_mul(self.input_neurons, self.Input_Hidden_weights) + self.Hidden_Layer_bias # Calcolo input hidden
        self.hidden_neurons = self.sigmoid(self.hidden_input) # Attivazione
        
        # l'input del neurone dell'Output riceve l'ouput dell'hidden come input. Successivamente applica attivazione
        self.output_input = matrix_mul(self.hidden_neurons, self.Hidden_Output_weights) + self.Output_Layer_bias
        self.output_neurons = self.softmax(self.output_input)

        return self.output_neurons

    # Serve per la derivazione per la backpropagation poiché
    # Esso ci dice quanto scostamento c'è tra l'output del neurone rispetto al suo input
    def sigmoide_derivate(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    # Discorso analogo del sigmoide
    def softmax_derivative(self, x):
        s = self.softmax(x)
        return s * (1 - s)

    # Propagazaione all'indietro, serve per allenare il modello
    # propaga l'errore all'indietro aggiustando i pesi, per minimizzare l'errore.
    def backpropagation(self, X, Y):
        
        output_error = self.output_neurons - Y # Differenza tra Previsione e Valore vero
        output_delta = output_error * self.softmax_derivative(self.output_input) # Errore tra Input - Output

        # Stesso discorso per Hidden (Però mentre torna indietro aggiungiamo i pesi del livello)
        hidden_error = matrix_mul(output_delta, self.Hidden_Output_weights.T) # Trasposizione, poiché torna all'indietro l'errore.
        hidden_delta = hidden_error * self.sigmoide_derivate(self.hidden_input) 

        # Aggiornamento della velocità e dei pesi
        # Utilizziamo la velocità per tenere traccia dei pesi passati
        self.Input_Hidden_velocity = self.momentum * self.Input_Hidden_velocity - self.learning_rate * matrix_mul(X.T, hidden_delta)  # Trasposta per indicare il senso opposto che deve andare + Passato l'errore dell'output di hidden.
        self.Hidden_Output_velocity = self.momentum * self.Hidden_Output_velocity - self.learning_rate * matrix_mul(self.hidden_neurons.T, output_delta) # Passato l'ouput di errore ritroso del n.output.

        # Aggiornamento dei layer della velocità
        self.Output_Layer_velocity = self.momentum * self.Output_Layer_velocity - self.learning_rate * np.sum(output_delta, axis=0, keepdims=True) 
        self.Hidden_Layer_velocity = self.momentum * self.Hidden_Layer_velocity - self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

        # Di conseguenza aggiorniamo i pesi.
        self.Input_Hidden_weights += self.Input_Hidden_velocity
        self.Hidden_Output_weights += self.Hidden_Output_velocity
        self.Hidden_Layer_bias += self.Hidden_Layer_velocity
        self.Output_Layer_bias += self.Output_Layer_velocity

    # Calcolo per vedere la differenza tra il nostro output e quello desiderato.
    # Usato principalmente per visualizzare il valore di scostamento
    def Mean_Squared_Error(self, Y_true, Y_pred):
        return np.mean(np.square(Y_true - Y_pred))