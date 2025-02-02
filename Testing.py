from Utils.utility import *

# Predict tramite dataset test
def TestingModel(mlp, X, Y, Types):

    Feature_predict = Output(mlp, X) # Tutte le probabilità generate dei feature
    #print(Feature_predict)
    
    Flower_predict = np.argmax(Feature_predict, axis=1) # Prende il max indice della riga [0.1, 0.7, 0.5] 
    #print(Flower_predict)
    
    Flower_list = np.argmax(Y, axis=1) # I veri valori presenti in formato [0, 1, 2]
    #print(Flower_list)
    
    for element in range(len(Feature_predict)):

        Confidence = (np.max(Feature_predict[element]) * 100).round(2) # Percentuale arrotondato tramite probabilità [0.1, 0.7, 0.5] 
        #print(Confidence)

        Predicted_Flower = Types[Flower_predict[element]] # Estrazione fiore in formato Stringa
        #print(Predicted_Flower)

        Current_Flower = Types[Flower_list[element]]
        
        if (Predicted_Flower == Current_Flower): # Comparazione tra le stringhe
            Result = "Corretto"
        else:
            Result = "Errato"

        print(f"Iterazione: [{element+1}] / Fiore Corrente: {Current_Flower} / Percentuale predetta: {Confidence}% / Esito: {Result}")

    Correct_Flowers = np.sum(Flower_predict == Flower_list) # Somma dei fiori corretti
    accuracy_tot = (Correct_Flowers / len(Y)) * 100 # Percentuale totale
    #print(Correct_Flower)
    #print(accuracy_tot)

    print("--- Risultato finale --- ")
    print(f"Accuracy del modello: {accuracy_tot:.2f}%")

    plt.figure(figsize=(8, 5))
    plt.bar(Types, accuracy_tot, color=['skyblue', 'green', 'violet'])
    plt.xlabel('Tipologia')
    plt.ylabel('Percentuale (%)')
    plt.savefig('Training-Model/Confidence.png')
    plt.show()


# Importazione
Feature, Flower = Load_Dataset("Dataset/test_set.txt")
Types = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

# Inizializzazione Modello
mlp = Model(Feature, Flower)
import_model(mlp, "Training-Model/IrisModel.json")

# Probabilistica delle soluzioni
TestingModel(mlp, Feature, Flower, Types)