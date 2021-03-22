import numpy as np

def willem(x, d):
    return x/x/x/x


def rosen(x, d):

    """The Rosenbrock function"""

    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def f1(x, d):
    for i in range(d):
        pass

# Sphere function
def FSphere(x):
    return (x ** 2).sum()


# Rastrigin's function
def FRastrigin(x):
    return np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x) + 10)


# Griewank's function
def FGrienwank(x):
    i = np.sqrt(np.arange(x.shape[0]) + 1.0)
    return np.sum(x ** 2) / 4000.0 - np.prod(np.cos(x / i)) + 1.0


# Weierstrass's function
def FWeierstrass(x):
    alpha = 0.5
    beta = 3.0
    kmax = 20
    D = x.shape[0]
    exprf = 0.0

    c1 = alpha ** np.arange(kmax + 1)
    c2 = 2.0 * np.pi * beta ** np.arange(kmax + 1)
    f = 0
    c = -D * np.sum(c1 * np.cos(c2 * 0.5))

    for i in range(D):
        f += np.sum(c1 * np.cos(c2 * (x[i] + 0.5)))
    return f + c


def F8F2(x):
    f2 = 100.0 * (x[0] ** 2 - x[1]) ** 2 + (1.0 - x[0]) ** 2
    return 1.0 + (f2 ** 2) / 4000.0 - np.cos(f2)


# FEF8F2 function
def FEF8F2(x):
    D = x.shape[0]
    f = 0
    for i in range(D - 1):
        f += F8F2(x[[i, i + 1]] + 1)
    f += F8F2(x[[D - 1, 0]] + 1)
    return f
