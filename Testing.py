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

X, Y = Load_Dataset("Dataset/test_set.txt")

mlp = Model(X, Y)

mlp.import_model("Training-Model/IrisModel.json")

accuracy = test(mlp, np.array(X), Y, debug=True)

print(f"Accuracy: {accuracy:.2f}%")
