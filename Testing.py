from Utils.utility import *

# Predict tramite dataset test
def testing_model(mlp, X, Y):

    Feature_predict = Output(mlp, X)
    print(Feature_predict)

    Flower_predict = np.argmax(Feature_predict, axis=1) # Prende il max indice della riga [0.1, 0.7, 0.5] 
    print(Flower_predict)
    
    Flower_list = np.argmax(Y, axis=1)
    print(Flower_list)

    Correct_Flower = np.sum(Flower_predict == Flower_list)
    print(Correct_Flower)

    accuracy = (Correct_Flower / Y.shape[0]) * 100
    print(accuracy)

    Types = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    
    for i, Feature_predict in enumerate(Feature_predict):
        percentage = (np.max(Feature_predict) * 100).round(2)
        Feature_predict = Types[Flower_predict[i]]
        actual = Types[Flower_list[i]]
        print(f"{percentage}% that it is {Feature_predict}. Actual: {actual}. Result: {Feature_predict == actual}")
    
    print(f"Test accuracy: {accuracy:.2f}%")

Feature, Flower = Load_Dataset("Dataset/test_set.txt")

mlp = Model(Feature, Flower)

import_model(mlp, "Training-Model/IrisModel.json")

testing_model(mlp, Feature, Flower)