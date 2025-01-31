from utility import *

def train(mlp, X, Y, epochs, debug=False):
    loss = np.zeros(epochs)

    for i in range(epochs):
        mlp.feedforward(X)
        mlp.backpropagation(X, Y)
        
        loss[i] = mlp.get_loss(Y, mlp.output_neurons)
        
    if debug:
        for i in range(epochs):
            print(f"Epoch {i+1}/{epochs}, Loss: {loss[i]:.2f}")

        plt.figure(figsize=(8, 5))
        epochs_range = np.arange(len(loss))

        plt.plot(epochs_range, loss, label='Loss', marker='o', markersize=4)
        plt.title('Loss in function of epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.show()

# Carica il dataset
X, Y = Load_Dataset("Dataset/training_set.txt")

# Crea e inizializza il modello MLP
mlp = Model(X, Y)

# Parametri di allenamento
epochs = 250

# Allena il modello
train(mlp, np.array(X), Y, epochs, debug=True)

# Esporta il modello allenato in un file JSON
mlp.export_model("Training-Model/IrisModel.json")
