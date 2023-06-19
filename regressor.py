import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Callable

from classifiers import sigmoid, linear
from losses import cross_entropy_loss
from optimizers import gradient_descent

class LogisticRegressor (object):
    def __init__ (self, classifier : Callable = sigmoid, 
                  loss : Callable = cross_entropy_loss, 
                  optimizer : Callable = gradient_descent, 
                  dimension : int = 2,
                  weights : np.ndarray = None):
        self.classifier = classifier
        self.loss = loss
        self.optimizer = optimizer
        self.weights = weights

        if weights == None:
            self.weights = np.zeros (dimension)

    def __call__(self, features : np.ndarray, targets : np.ndarray, epochs = 100) -> None:
        features = np.c_[features, np.ones(features.shape[0])]

        for _ in range(epochs):
            for i, (x,y) in enumerate(zip(features, targets)):
                z = np.dot(self.weights, x)
                y_pred = self.classifier(z)
                loss = self.loss(y_pred, y)
                
                print(f"{i} / {targets.shape[0]}")
                print(f"pred : {y_pred}")
                print(f"loss : {loss}")
                print(f"w8s  : {self.weights}")

                self.weights = gradient_descent (self.weights, x,y,y_pred)


if __name__ == "__main__":
    print("testing logistic regressor")
    lr = LogisticRegressor (classifier=sigmoid)
    df = pd.read_csv("logistic_dummy.csv")
    target = df.pop("y").to_numpy()
    features = df.to_numpy()

    lr(features, target)
    y = []
    features = np.c_[features, np.ones(features.shape[0])]
    for x in features:
        y.append(lr.classifier(np.dot(lr.weights, x)))
    plt.subplot(1,2,1)
    plt.title("trained regressor")
    plt.plot(np.column_stack(features)[0],y)
    plt.plot(np.column_stack(features)[0],target,"o")


    print("testing linear regressor")
    lr = LogisticRegressor (classifier=linear)
    df = pd.read_csv("linear_dummy.csv")
    target = df.pop("y").to_numpy()
    features = df.to_numpy()

    lr(features, target)
    y = []
    features = np.c_[features, np.ones(features.shape[0])]
    for x in features:
        y.append(lr.classifier(np.dot(lr.weights, x)))
    plt.subplot(1,2,2)
    plt.title("trained regressor")
    plt.plot(np.column_stack(features)[0],y)
    plt.plot(np.column_stack(features)[0],target,"o")
    plt.show()
