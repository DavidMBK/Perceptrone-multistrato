from dataset import Load_Dataset
from MLPNetwork import MLP
import numpy as np

# Carica il dataset
X, Y = Load_Dataset("Dataset/test_set.txt")

# Converti Y in one-hot encoding
Y_one_hot = np.zeros((len(Y), 3))
for i, y in enumerate(Y):
    Y_one_hot[i, y] = 1

# Crea il modello MLP
input_size = len(X[0])
hidden_size = 5
output_size = 3
learning_rate = 0.01
mlp = MLP(input_size, hidden_size, output_size, learning_rate)

# Importa il modello
mlp.import_model("model.json")

# Testa il modello
accuracy = mlp.test(np.array(X), Y_one_hot, debug=True)
print(f"Accuracy: {accuracy:.2f}%")