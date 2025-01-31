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
    
def train_and_test(mlp, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, epochs: int, debug: bool=False) -> None:
        train_loss = np.zeros(epochs)
        test_loss = np.zeros(epochs)

        for i in range(epochs):
            # Training step
            mlp.feedforward(X_train)
            mlp.backpropagation(X_train, y_train)
            train_loss[i] = mlp.error_function(y_train, mlp.output_neurons)

            # Test step
            test_predictions = mlp.feedforward(X_test)
            test_loss[i] = mlp.error_function(y_test, test_predictions)
            
        if debug:
            for i in range(epochs):
                print(f"Epoch {i+1}/{epochs}, Training loss: {train_loss[i]:.4f}, Test loss: {test_loss[i]:.4f}")

            plt.figure(figsize=(10, 6))
            epochs_range = np.arange(epochs)

            plt.plot(epochs_range, train_loss, label='Training loss', marker='o', markersize=4)
            plt.plot(epochs_range, test_loss, label='Test loss', marker='x', markersize=4)
            plt.title('Training and test loss over epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            plt.show()

X, Y = Load_Dataset("Dataset/test_set.txt")

mlp = Model(X, Y)

import_model(mlp, "Training-Model/IrisModel.json")

accuracy = test(mlp, np.array(X), Y, debug=True)

print(f"Accuracy: {accuracy:.2f}%")

epochs = 100

train_and_test(mlp, np.array(X), Y, np.array(X), Y, epochs, debug=True)