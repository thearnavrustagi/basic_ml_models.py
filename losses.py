import numpy as np

def cross_entropy_loss (y_pred, y):
    return - (y * np.log(y_pred) +  (1-y)*np.log(1-y_pred))

if __name__ == "__main__":
    y_pred, y = np.array([0.98,0.4,0.1,0.8]), np.array([1,0,0,0])
    print("checking cross entropy loss")
    print(f"y_pred : {y_pred}")
    print(f"y      : {y}")
    print(f"loss   : {cross_entropy_loss(y_pred, y)}")
