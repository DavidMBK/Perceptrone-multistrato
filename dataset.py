def Load_Dataset(file_path):
        training_dataset = open(file_path, "r")
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
                if classes == "Iris-versicolor":
                        Y[Y.index(classes)] = 1
                if classes == "Iris-virginica":
                        Y[Y.index(classes)] = 2

        return X, Y  


