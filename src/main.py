import time

import numpy as np
from sklearn.metrics import accuracy_score

from src import data_set, kernels
from src.classifier import Classifier

_train_ratio = 0.8

if __name__ == '__main__':
    start = time.time()
    steps = 40
    c = 500
    lr = 0.01
    kernel = kernels.laplace_rbf

    ###########################
    # Data Set Load
    ###########################
    x_set = data_set.read_x()
    y_set = data_set.read_y()

    ###########################
    # Training & Cross Check Sets
    ###########################
    _train_index = int(len(x_set) * _train_ratio)
    y_train, y_test = y_set[:_train_index], y_set[_train_index:]
    x_train, x_test = x_set[:_train_index, :], x_set[_train_index:, :]

    ###########################
    # Training
    ###########################
    classifier = Classifier(c=c, lr=lr, steps=steps, kernel=kernel)
    classifier.fit(x_train, y_train)

    end = time.time()
    print(end - start)
    ###########################
    # Cross Check
    ###########################
    y_predict = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy for cross check: {accuracy}")

    ###########################
    # Prediction
    ###########################
    x_tpi = data_set.read_x_test()
    y_tpi = classifier.predict(x_tpi)
    np.savetxt("prediction/Y.csv", y_tpi, delimiter=",")

    ###########################
    # Weight Persistence
    ###########################
    np.savetxt("W.csv", classifier.weights, delimiter=",")
