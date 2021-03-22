import numpy as np

def willem(x, d):
    return x/x/x/x


def rosen(x, d):

    """The Rosenbrock function"""

    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def f1(x, d):
    for i in range(d):
        pass
