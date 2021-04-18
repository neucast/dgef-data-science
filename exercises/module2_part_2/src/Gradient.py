import numpy as np

"""Compute the value of P1 = (x1, y1)

Parameters
----------
Dfx :   function
        Partial derivative of f(x) respect x.
      
Dfy :   function
        Partial derivative of f(x) respect y.
      
x0 :    real number
        value of the abscissa x.
    
y0 :    real number
        value of the ordinate
Returns
-------
P1 :    point (x1, y1)
        Implement the function P1 = P0 - 0.5 * Gradientf(P0).

Examples
--------
>>> f = lambda x, y: x ** 2 - y ** 2 + 1
>>> Dfx = lambda x: 2 * x
>>> Dfy = lambda y: -2 * y
>>> x0 = 1
>>> y0 = 1
>>> P1 = compute_P1(Dfx, Dfy, x0, y0)
P1 = (0, 2)
"""


def compute_P1(Dfx, Dfy, x0, y0) -> object:
    P0 = np.array([x0, y0])
    Gradientf = lambda P: np.array([Dfx(P[0]), Dfy(P[1])])
    P1 = P0 - 0.5 * Gradientf(P0)
    return P1
