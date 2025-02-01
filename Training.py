from Utils.utility import *

def combine(mlp, X, Y, iterazione):
    loss_history = train(mlp, X, Y, iterazione)
    visualizze_graph(loss_history)
    return export_model(mlp, "Training-Model/IrisModel.json")
    
def visualizze_graph(loss_history):
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

    return np.save("Training-Model/train_graph.npy", loss_history)  # Salva la loss in un file

def train(mlp, X, Y, iterazione):
    # Inizializzazione del vettore di perdita
    loss_history = np.zeros(iterazione)

    for i in range(iterazione):
        mlp.feedforward(X) # Calcolo OutPut
        mlp.backpropagation(X, Y) # Aggiornamento dei Pesi
        loss_history[i] = mlp.Mean_Squared_Error(Y, mlp.output_neurons) # Calcolo della perdita
        print(f"Iterazione: [{i+1}/{iterazione}], Perdita: {loss_history[i]:.2f}")

    return loss_history

# Inizializzazione del modello
X, Y = Load_Dataset("Dataset/training_set.txt")
mlp = Model(X, Y)

iterazione = get_iterazioni()

# Allenamento ed export del modello in formato json.
combine(mlp, np.array(X), Y, iterazione)


