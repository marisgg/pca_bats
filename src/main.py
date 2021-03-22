#!/usr/bin/env python3
import numpy as np
from bat import Bat
import functions

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
    algorithm = Bat(
        # Dimension
        1,
        # Population
        40,
        # Generations       
        1000,
        # Loudness  
        0.5,
        # Pulse rate
        0.5,
        # Min. Freq.
        0.0,
        # Max. Freq.
        2.0,
        # Lower bound
        -30.0,
        # Upper bound
        30.0,
        # functions.rosen
        function
        )
    print(algorithm.run_bats())

if __name__ == "__main__":
    main()