from utility import *

def display_results():
    return null

def test(mlp, X, Y):
    predictions = mlp.feedforward(X)
    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = np.argmax(Y, axis=1)

    correct_predictions = np.sum(predicted_classes == actual_classes)
    accuracy = (correct_predictions / Y.shape[0]) * 100

    species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
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
    
def train_and_test(mlp, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, iterazione: int, debug: bool=False) -> None:
        train_loss = np.zeros(iterazione)
        test_loss = np.zeros(iterazione)

        for i in range(iterazione):
            # Training step
            mlp.feedforward(X_train)
            mlp.backpropagation(X_train, y_train)
            train_loss[i] = mlp.Mean_Squared_Error(y_train, mlp.output_neurons)

            # Test step
            test_predictions = mlp.feedforward(X_test)
            test_loss[i] = mlp.Mean_Squared_Error(y_test, test_predictions)
            
        if debug:
            for i in range(iterazione):
                print(f"Epoch {i+1}/{iterazione}, Training loss: {train_loss[i]:.4f}, Test loss: {test_loss[i]:.4f}")

            plt.figure(figsize=(10, 6))
            iterazione_range = np.arange(iterazione)

            plt.plot(iterazione_range, train_loss, label='Training loss', marker='o', markersize=4)
            plt.plot(iterazione_range, test_loss, label='Test loss', marker='x', markersize=4)
            plt.title('Training and test loss over iterazione')
            plt.xlabel('iterazione')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            plt.show()

X, Y = Load_Dataset("Dataset/test_set.txt")

mlp = Model(X, Y)

import_model(mlp, "Training-Model/IrisModel.json")

accuracy = test(mlp, np.array(X), Y)

print(f"Accuracy: {accuracy:.2f}%")

iterazione = get_iterazioni()

train_and_test(mlp, np.array(X), Y, np.array(X), Y, iterazione, debug=True)