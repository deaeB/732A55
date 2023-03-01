
import numpy as np
from matplotlib import pyplot as plt


def getpolicy(Q):
    """ Get best policy matrix from the Q-matrix.
    You have to implement this function yourself. It is not necessary to loop
    in order to do this, and looping will be much slower than using matrix
    operations. It's possible to implement this in one line of code.
    """
    # Q(s(Y,X),a)

    P = np.argmax(Q, axis = 2)
    # P>(Y,X)
    return P


def getvalue(Q):
    """ Get best value matrix from the Q-matrix.
    You have to implement this function yourself. It is not necessary to loop
    in order to do this, and looping will be much slower than using matrix
    operations. It's possible to implement this in one line of code.
    """
    # Q(s(Y,X),a)
    Y,X,a = Q.shape
    P = np.argmax(Q, axis = 2)
    V = []
    
    for x in range(X):
        for y in range(Y):
            V.append(Q[y,x,P[y,x]])
    
    V = np.array(V).reshape((Y,X))

    return V

