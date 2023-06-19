import numpy as np
import matplotlib.pyplot as plt


def sigmoid (z):
    return 1 / (1 + np.exp(-z))

def linear (z):
    return z

def __plot__ (x, func):
    y = func(x)
    plt.plot(x,y)

classifiers = [
            (sigmoid, "sigmoid"),
            (linear, "linear"),
        ]
classifiers = np.column_stack(classifiers)

if __name__ == "__main__":
    x = np.linspace (-5,5,100)
    for classifier in classifiers[0]:
        __plot__(x, classifier)

    plt.legend(classifiers[1])
    plt.show()
