#!/usr/bin/env python3
import numpy as np
from bat import Bat

popsize = 100
dimension = np.array([100, 500, 1000])
run_times = 25
maxFEs = 5000 * dimension

seed = 0

np.random.seed(seed)

def function(x, dimension):
    value = 0
    for i in range(dimension):
        value += np.sum((-x[i] * np.sin(np.sqrt(np.abs(x[i])))))
    return value

def main():
    rng = np.random.default_rng(seed)
    algorithm = Bat(1, 40, 1000, 0.5, 0.5, 0.0, 2.0, -10.0, 10.0, function)
    print(algorithm.run_bats())

if __name__ == "__main__":
    main()