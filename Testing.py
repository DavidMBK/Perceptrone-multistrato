from Utils.utility import *

# Predict tramite dataset test
def TestingModel(mlp, X, Y, Types):

    Feature_predict = Output(mlp, X) # Tutte le probabilità generate dai feature [0.1, 0.7, 0.2]
    #print(Feature_predict)
    
    Flower_predict = np.argmax(Feature_predict, axis=1) # Prende il max indice della riga [0.1, 0.7, 0.2] = [1]
    #print(Flower_predict)
    
    Flower_list = np.argmax(Y, axis=1) # I veri valori presenti in formato [0, 1, 2]
    #print(Flower_list)
    
    for element in range(len(Feature_predict)):

        Confidence = (np.max(Feature_predict[element]) * 100).round(2) # Percentuale arrotondato tramite probabilità [0.1, 0.7, 0.5] = 70% 
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
    Accuracy_tot = (Correct_Flowers / len(Y)) * 100 # Percentuale totale di tutti i fiori corretti rispetto alla quantità
    #print(Correct_Flower)
    #print(Accuracy_tot)

    if (Accuracy_tot == 100):
        Ending = "Il Modello ha imparato perfettamente"
    else:
        Ending = "Il Modello non ha imparato perfettamente"

    print("--- Risultato finale --- ")
    print(f"Accuracy del modello: {Accuracy_tot:.2f}% - {Ending}")

    plt.figure(figsize=(8, 5))
    plt.bar(Types, Accuracy_tot, color=['skyblue', 'green', 'violet']) # Colore dei pilastri
    plt.xlabel('Tipologia')
    plt.ylabel('Percentuale (%)')
    plt.savefig('Training-Model/Confidence.png')
    plt.show()


# Importazione
Feature, Flower = Load_Dataset("Dataset/test_set.txt")

# Inizializzazione Modello
mlp = Model(Feature, Flower)
import_model(mlp, "Training-Model/IrisModel.json")

# Probabilistica delle soluzioni
TestingModel(mlp, Feature, Flower, GetTypes())

## Cosa si potrebbe aggiungere? Un grafico per mostrare le aree di successo? 