from utility import *

def test(mlp, X, y, debug=False):
    predictions = mlp.feedforward(X)
    
    # get the index of the highest probability for each row
    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = np.argmax(y, axis=1)

    # get the number of correct predictions and calculate the accuracy
    correct_predictions = np.sum(predicted_classes == actual_classes)
    accuracy = (correct_predictions / y.shape[0]) * 100

    if debug:
        species = ["Iris-setosa", "Iris-versicOutput_Layeror", "Iris-virginica"]
        
        # print the predictions and the actual classes for each row
        for i, prediction in enumerate(predictions):
            percentage = (np.max(prediction) * 100).round(2)
            predicted = species[predicted_classes[i]]
            actual = species[actual_classes[i]]
            print(f"{percentage}% that it is {predicted}. Actual: {actual}. Result: {predicted == actual}")
        
        print(f"Test accuracy: {accuracy:.2f}%")

    return accuracy

def import_model(mlp, file_path):
    with open(file_path, 'r') as file:
        IrisDataModel = json.load(file)
    
    mlp.input_size = IrisDataModel["input_size"]
    mlp.hidden_size = IrisDataModel["hidden_size"]
    mlp.output_size = IrisDataModel["output_size"]
    mlp.learning_rate = IrisDataModel["learning_rate"]
    mlp.momentum = IrisDataModel["momentum"]

    mlp.Input_Hidden_weights = np.array(IrisDataModel["Weights"]["Input_Hidden_weights"])
    mlp.Hidden_Output_weights = np.array(IrisDataModel["Weights"]["Hidden_Output_weights"])

    mlp.Hidden_Layer_bias = np.array(IrisDataModel["Bias"]["Hidden_Layer_bias"])
    mlp.Output_Layer_bias = np.array(IrisDataModel["Bias"]["Output_Layer_bias"])
    
    mlp.Input_Hidden_velocity = np.array(IrisDataModel["Momentum"]["Input_Hidden_velocity"])
    mlp.Hidden_Output_velocity = np.array(IrisDataModel["Momentum"]["Hidden_Output_velocity"])
    mlp.Hidden_Layer_velocity = np.array(IrisDataModel["Momentum"]["Hidden_Layer_velocity"])
    mlp.Output_Layer_velocity = np.array(IrisDataModel["Momentum"]["Output_Layer_velocity"])

    print(f"Model imported from {file_path}")

X, Y = Load_Dataset("Dataset/test_set.txt")

mlp = Model(X, Y)

import_model(mlp, "Training-Model/IrisModel.json")

accuracy = test(mlp, np.array(X), Y, debug=True)

print(f"Accuracy: {accuracy:.2f}%")
