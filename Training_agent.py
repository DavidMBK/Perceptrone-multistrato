# Recupero dei dati

training_dataset = open("Dataset/training_set.txt", "r")

X = [] # Carattersitiche
Y = [] # Target

for line in training_dataset:
        splitted_line = line.strip().split(",")
        features = list(map(float, splitted_line[:-1]))  
        target = splitted_line[-1]
    
        X.append(features)  
        Y.append(target)  

print("Prime 2 righe di X:", X[:2])
print("Prime 2 righe di y:", Y[:2])

