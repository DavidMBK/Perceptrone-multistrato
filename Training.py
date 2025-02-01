from utility import *

def train(mlp, X, Y, iterazione):
    # Inizializzazione del vettore di perdita
    loss_history = np.zeros(iterazione)

    for i in range(iterazione):
        mlp.feedforward(X) # Calcolo OutPut
        mlp.backpropagation(X, Y) # Aggiornamento dei Pesi
        loss_history[i] = mlp.Mean_Squared_Error(Y, mlp.output_neurons) # Calcolo della perdita
        print(f"Iterazione: [{i+1}/{iterazione}], Perdita: {loss_history[i]:.2f}")

    iterazione_range = np.arange(len(loss_history)) # Asse X per le iterazioni
    
    plt.style.use("cyberpunk")
    plt.figure(figsize=(12, 8))
    plt.plot(iterazione_range, loss_history, label='Perdita', marker='o', markersize=4)
    plt.title('Andamento del Training', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Iterazione', fontsize=14, fontweight='bold')
    plt.ylabel('Perdita (Loss)', fontsize=14, fontweight='bold')
    plt.grid()
    plt.legend()
    mplcyberpunk.add_glow_effects()
    plt.show()

    np.save("Training-Model/train_graph.npy", loss_history)  # Salva la loss in un file

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

# Inizializzazione del modello
X, Y = Load_Dataset("Dataset/training_set.txt")
mlp = Model(X, Y)

iterazione = get_iterazioni()

# Allenamento ed export del modello in formato json.
train(mlp, np.array(X), Y, iterazione)
export_model(mlp, "Training-Model/IrisModel.json")
