import numpy as np
import matplotlib.pyplot as plt
from sympy import var,symbols, init_printing, diff, linear_eq_to_matrix, solve_linear, Eq

from functools import reduce

def line (initial, data):
class LinearRegressor(object):
    """
    shape of points is (2,N), 
    0th index has all X cooridinates
    1st index has all Y coordinates
    """
    def __init__ (self, features : np.ndarray, fiting=line , error=lambda x : x**2, ) :
        # this should be [ [features ... , target ], ] 
        self.labels = features
        self.feature_dimension = self.labels.shape[1] - 1
        self.error = error

        # index 0 is slope
        # index 1 is intercept
        self.best_fit_line = (None,None)

    def linear_regression (self) -> tuple :
        # m0 is the constant term
        symbol_names = [var(f'm{n}') for n in range(self.feature_dimension+1)]
        feature_names = [symbols(f'x{n+1}') for n in range(self.feature_dimension)]
        target_symbol = symbols('y')

        eqn = [var*symbol for symbol,var in zip(symbol_names[1:],feature_names)]
        eqn = self.error(symbol_names[0] + reduce(lambda a,b : a+b, eqn) - target_symbol)
        print(eqn)
        
        eqns = []
        for symbol in symbol_names:
            eqns.append(diff(eqn,symbol))

        print(eqns)
        substitutor = {}
        columns = np.column_stack(self.labels)
        for i,feature in enumerate(feature_names):
            substitutor[str(feature)] = sum(columns[i])
        substitutor[str(target_symbol)] = sum(columns[-1])

        for key, val in substitutor.items():
            for i,eqn in enumerate(eqns):
                eqns[i] = eqn.subs(var(key),val)
        print(eqns)
        print(*symbol_names)
        mat = linear_eq_to_matrix(eqns,symbol_names)
        print(mat)
        lhs,rhs = tuple((np.array(m).astype(np.float64)) for m in mat)

        print(np.linalg.solve(lhs, rhs))
            



if __name__ == "__main__":
    print("testing linear regressor")
    regressor = LinearRegressor(np.array(np.column_stack([[1,2,3,4,5,6],[0,1,2,2,2,3]])))
    points = regressor.linear_regression()
    m,c = points[0], points[1]

    
    plt.plot(regressor.points[0], regressor.points[1],"o")
    plt.plot([0,6],[c,m*6+c])
    plt.show()
