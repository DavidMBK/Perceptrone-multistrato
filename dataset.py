def Load_Dataset():
        training_dataset = open("Dataset/training_set.txt", "r")
        X = [] # Carattersitiche
        Y = [] # Target

        for line in training_dataset:
                splitted_line = line.strip().split(",")
                features = list(map(float, splitted_line[:-1]))  
                target = splitted_line[-1]
        
                X.append(features)  
                Y.append(target)

        for classes in Y:
                if classes == "Iris-setosa":
                        Y[Y.index(classes)] = 0
                elif classes == "Iris-versicolor":
                        Y[Y.index(classes)] = 1
                elif classes == "Iris-virginica":
                        Y[Y.index(classes)] = 2

        return X, Y  


