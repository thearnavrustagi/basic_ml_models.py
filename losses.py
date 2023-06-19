import numpy as np

def cross_entropy_loss (y_pred, y):
    return - (y * np.log(y_pred) +  (1-y)*np.log(1-y_pred))

def entropy_loss (y : np.ndarray):
    values = set(y)
    total  = y.shape[0]
    net_entropy = 0
    for value in values:
        p = np.count_nonzero(y == value) / total
        net_entropy += -(p * np.log2(p))

    return net_entropy

def conditional_entropy_loss (x : np.ndarray, y : np.ndarray):
    values = set(x)
    total = x.shape[0]
    net_entropy = 0
    for value in values:
        indexes = np.where(x == value)
        p = np.count_nonzero(x == value) / total
        e = entropy_loss(np.array([y[i] for i in indexes])[0])
        net_entropy += p * e

    return net_entropy

if __name__ == "__main__":
    y_pred, y = np.array([0.98,0.4,0.1,0.8]), np.array([1,0,0,0])
    print("checking cross entropy loss")
    print(f"y_pred : {y_pred}")
    print(f"y      : {y}")
    print(f"loss   : {cross_entropy_loss(y_pred, y)}")

    y = np.array([0,0,1,1,1,0,0,1,1,1,1,1,1,0])
    print("checking entropy loss")
    print(f"y      : {y}")
    print(f"loss   : {entropy_loss(y)}")
    
    x = np.array([1,1,1,1,0,0,0,1,0,0,0,1,0,1])
    print(f"checking conditional entropy loss")
    print(f"x       : {x}")
    print(f"y       : {y}")
    print(f"gain    : {conditional_entropy_loss(x,y)}")

