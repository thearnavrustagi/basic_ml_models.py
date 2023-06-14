import numpy as np

from losses import cross_entropy_loss

"""
PARAMETERS
w : weights of the given regression model
x : the feature label for a given test case
y : the actual target label of the given test case

y_pred : the predicted target label of the given test case

learning_rate : the step length of the optimizer

RETURNS
the updated weights
"""
def gradient_descent (w, x, y, y_pred, learning_rate=1e-2):
    # here we are finding the gradient of the given regression function
    diff = ( y_pred - y ) * x

    # changing the weights based on the learning rate
    w -= learning_rate * diff
    return w

if __name__ == "__main__":
    x = np.array([3.,2.,1.])
    w = np.array([0.,0.,0.])
    y = 1
    y_pred = 0.5
    print("testing gradient descent")
    print("example from https://web.stanford.edu/~jurafsky/slp3/5.pdf page 17")
    print("expected answer : [.15, .1, .05]")
    print(f"changed weights {gradient_descent(w,x,y,y_pred,0.1)}")
    
