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
    print(f"Model exported to {file_path}")

# Carica il dataset
X, Y = Load_Dataset("Dataset/training_set.txt")

# Crea e inizializza il modello MLP
mlp = Model(X, Y)

# Parametri di allenamento
epochs = 1000

# Allena il modello
train(mlp, np.array(X), Y, epochs, debug=True)

# Esporta il modello allenato in un file JSON
export_model(mlp, "Training-Model/IrisModel.json")
