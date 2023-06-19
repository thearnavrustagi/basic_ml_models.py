import numpy as np
import pandas as pd

from typing import Callable
from losses import entropy_loss, conditional_entropy_loss

class __cond__ (object):
    def __init__ (self, i, comp):
        self.index = i
        self.comparator = comp

    def __call__ (self,x):
        return self.comparator == x[self.index]

class __node__(object):
    def __init__ (self, 
                  condition : Callable, 
                  loss : Callable, 
                  conditional_loss : Callable):
        self.condition = condition
        self.children = []

        self.loss = loss
        self.conditional_loss = conditional_loss

        self.prediction = None

    def satisfy (self,x):
        return self.condition(x)

    def __call__ (self,x,y):
        y_entropy = self.loss(y)
        information_gain = []

        x = np.column_stack(x)
        for i,feature in enumerate(x):
            gain = (y_entropy - self.conditional_loss(feature,y))
            information_gain.append((i,gain))

        information_gain = sorted(information_gain, key = lambda x : x[1])
        if information_gain[-1][1] == 0: 
            self.__create_inference_node__(y[0])
            return
        feature = information_gain[-1][0]

        for val in set(x[feature]):
            node = __node__(__cond__(feature, val),self.loss, self.conditional_loss)
            xnew, ynew = [], []
            for i,e in enumerate(np.column_stack(x)):
                if e[feature] == val:
                    xnew.append(e)
                    ynew.append(y[i])
            xnew, ynew = np.array(xnew), np.array(ynew)
            node(xnew, ynew)
            self.children.append(node)

    def __create_inference_node__(self, y):
        self.prediction = y

class DecisionTree(object):
    def __init__ (self, 
                  loss : Callable = entropy_loss, 
                  conditional_loss : Callable = conditional_entropy_loss):
        self.loss = loss
        self.conditional_loss = conditional_loss

        self.root = __node__(lambda x : True, loss, conditional_loss)

    def __call__ (self,x,y):
        self.root(x,y)

    def traverse (self,x):
        itr = self.root

        while len(itr.children) != 0:
            for child in itr.children:
                if child.condition(x):
                    itr = child
                    break
        return itr.prediction

if __name__ == "__main__":
    tree = DecisionTree()

    df = pd.read_csv("decisiontree.csv")
    df.pop("car name")
    y = df.pop("mpg").to_numpy()
    x = df.to_numpy()

    tree(x,y)
    print("""testing decision tree\ngiving first row of dataset\nexpected ans = 18""")
    print(tree.traverse([8,307,130,3504,12,70,1]))

