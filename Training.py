from Utils.utility import *

def combine(mlp, iterazione, Feature_train, Target_train, Feature_test, Target_test): 
    MSE_train, MSE_test = ModelTraining(mlp, iterazione, Feature_train, Target_train, Feature_test, Target_test)
    visualizze_graph(iterazione, MSE_train, MSE_test)
    return export_model(mlp, "Training-Model/IrisModel.json")
    
def visualizze_graph(iterazione, train_loss, test_loss):
    plt.style.use("cyberpunk") # import mplcyberpunk : https://github.com/dhaitz/mplcyberpunk
    plt.figure(figsize=(12, 8))
    
    #print(iterazione)
    iterazione_range = list(range(iterazione)) # 3 ad [0,1,2] per il grafico
    #print(iterazione_range)

    plt.plot(iterazione_range, train_loss, label='Perdità in Training', marker='o', markersize=4) # Linea di train
    plt.plot(iterazione_range, test_loss, label='Perdità in Testing', marker='x', markersize=4) # Linea di test
    plt.title('Errore totale su Training e Test', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('iterazione', fontsize=14, fontweight='bold')
    plt.ylabel('Perdita', fontsize=14, fontweight='bold')
    plt.grid()
    plt.legend()
    mplcyberpunk.add_glow_effects() # Effetto glow importato dalla libreria
    plt.savefig('Training-Model/Total_Error_Graph.png')
    plt.show()

def ModelTraining(mlp, iterazione, Feature_train, Target_train, Feature_test, Target_test):
        MSE_train = np.zeros(iterazione) # Perdita derivante dal trainng dataset
        MSE_test = np.zeros(iterazione) # Perdita derivante dal test dataset

        for i in range(iterazione):
            train_output = Output(mlp, Feature_train) # Si ottiene l'ouput predetto tramite feedforward
            train_errorprop = mlp.backpropagation(Feature_train, Target_train) # Alleniamo il modello e aggiorniamo i pesi
            MSE_train[i] = mlp.Mean_Squared_Error(Target_train, train_output) # Calcolo del discostamento tra output previsto e output generato
            # Idem qua
            test_predictions = Output(mlp, Feature_test) 
            MSE_test[i] = mlp.Mean_Squared_Error(Target_test, test_predictions)
            print(f"Iterazione {i}: Training Loss = {MSE_train[i]}, Test Loss = {MSE_test[i]}")
        print(f"Iter 0 - Training Loss: {MSE_train[0]}, Test Loss: {MSE_test[0]}")
        print(f"Iter {iterazione-1} - Training Loss: {MSE_train[-1]}, Test Loss: {MSE_test[-1]}")


        return MSE_train, MSE_test

# Recupero dei dati
# Feature = le caratteristiche del fiore iris, Sepal width/height e Petal width/height.
# Target = la tipologia del fiore, ovvero che tipo di fiore è Setosa etc.
Feature_train, Target_train = Load_Dataset("Dataset/training_set.txt")
Feature_test, Target_test = Load_Dataset("Dataset/test_set.txt")

#Inizializzazione del modello 
mlp = Model(Feature_train, Target_train) 

# Allenamento, grafico ed export del modello in formato json.
combine(mlp, get_iterazioni(), Feature_train, Target_train, Feature_test, Target_test)


