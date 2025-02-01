from Utils.utility import *

def display_results(iterazione, train_loss, test_loss):
    plt.style.use("cyberpunk")
    plt.figure(figsize=(12, 8))
    iterazione_range = np.arange(iterazione)

    plt.plot(iterazione_range, train_loss, label='Training loss', marker='o', markersize=4)
    plt.plot(iterazione_range, test_loss, label='Test loss', marker='x', markersize=4)
    plt.title('Errore totale su Training e Test', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('iterazione', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.grid(True)
    plt.legend()
    mplcyberpunk.add_glow_effects()
    plt.show()


def testing_model(mlp, X, Y):
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
        # print(f"{percentage}% that it is {predicted}. Actual: {actual}. Result: {predicted == actual}")
    print(f"Test accuracy: {accuracy:.2f}%")

    return accuracy

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
    
        display_results(iterazione, train_loss, test_loss)

X_train, Y_train = Load_Dataset("Dataset/training_set.txt")
X_test, Y_test = Load_Dataset("Dataset/test_set.txt")

mlp = Model(X_train, Y_train)

# import_model(mlp, "Training-Model/IrisModel.json")

accuracy_test = testing_model(mlp, np.array(X_test), Y_test) # 33% 

iterazione = get_iterazioni()

train_and_test(mlp, np.array(X_train), Y_train, np.array(X_test), Y_test, iterazione, debug=True) # Lo sta riallenando ma io c'è l'ho già allenato. 
# Voglio usare il mio modello e testare e basta...

accuracy_test = testing_model(mlp, np.array(X_test), Y_test)  # 96%
