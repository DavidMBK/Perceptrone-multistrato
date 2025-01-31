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
    model_data = {
        "input_size": mlp.input_size,
        "hidden_size": mlp.hidden_size,
        "output_size": mlp.output_size,
        "learning_rate": mlp.learning_rate,
        "momentum": mlp.momentum,
        "IH_weights": mlp.IH_weights.tolist(),
        "HO_weights": mlp.HO_weights.tolist(),
        "HL_bias": mlp.HL_bias.tolist(),
        "OL_bias": mlp.OL_bias.tolist(),
        "IH_velocity": mlp.IH_velocity.tolist(),
        "HO_velocity": mlp.HO_velocity.tolist(),
        "HL_velocity": mlp.HL_velocity.tolist(),
        "OL_velocity": mlp.OL_velocity.tolist()
    }
    
    with open(file_path, 'w') as file:
        json.dump(model_data, file, indent=4)
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
