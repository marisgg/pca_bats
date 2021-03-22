#!/usr/bin/env python3
import numpy as np
from bat import Bat
import functions

popsize = 100
dimension = np.array([100, 500, 1000])
run_times = 25
maxFEs = 5000 * dimension

def function(x, dimension):
    value = 0
    for i in range(dimension):
        value += np.sum((-x[i] * np.sin(np.sqrt(np.abs(x[i])))))
    return value

def main():
    algorithm = Bat(
        # Dimension
        2,
        # Population
        400,
        # Generations       
        10000,
        # Loudness  
        0.5,
        # Pulse rate
        0.5,
        # Min. Freq.
        0.0,
        # Max. Freq.
        100.0,
        # Lower bound
        -1,
        # Upper bound
        1,
        functions.rosen
        # functions.willem
        )
    print(algorithm.run_bats())

if __name__ == "__main__":
    main()